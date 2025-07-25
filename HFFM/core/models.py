import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViltProcessor, ViltForQuestionAnswering
from torch.optim import AdamW
from torch.utils.data import Subset
import random
from tqdm.auto import tqdm
import numpy as np
import torch.nn as nn
from transformers import ViltModel 
from collections import Counter
import copy
import re
import gc
from torch.nn import Parameter
import torch.nn.functional as F


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.act       = nn.GELU()
        self.up_proj   = nn.Linear(bottleneck_size, hidden_size)
        # self.norm = nn.LayerNorm(hidden_size)
    def forward(self, x):
        return  self.up_proj(self.act(self.down_proj(x)))

class DoubleAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size):
        super().__init__()
        # first adapter
        self.ad1 = Adapter(hidden_size, bottleneck_size)
        self.ad2 = Adapter(hidden_size, bottleneck_size)
        # self.ad13 = Adapter(hidden_size, bottleneck_size)
        # second adapter
        # self.ad2 = Adapter(hidden_size, bottleneck_size)
        # self.ad22 = Adapter(hidden_size, bottleneck_size)
        # self.alpha = nn.Parameter(torch.tensor(1.0))
        # self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # a = torch.sigmoid(self.alpha)  
        # x → adapter1 → adapter2, each with its own residual
        h =self.ad1(x)+self.ad2(x)
        return h



class ViltWithCustomClassifier(torch.nn.Module):
    def __init__(self, base_model: ViltModel):
        super().__init__()
        self.vilt = base_model
        for param in self.vilt.parameters():
            param.requires_grad = False
        self.hidden_size = base_model.config.hidden_size
        # self.classifier_heads = nn.ModuleList([nn.Linear(hidden_size, num_label) for num_label in num_labels])

        self.classifier = None

    def set_classifier(self, classifier: nn.Module):
        """Attach one of your external heads to this model."""
        self.classifier = classifier
        self.classifier.to(device)
        # make sure its parameters will be trained
        for p in self.classifier.parameters():
            p.requires_grad = True

    def forward(self,**inputs):
        assert self.classifier is not None
        outputs = self.vilt(**inputs)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


def add_adapters_to_vilt(model, bottleneck=128):
    for layer in model.vilt.encoder.layer:
        # stash original
        if not hasattr(layer.output, "_orig_forward"):
            layer.output._orig_forward = layer.output.forward

        # build a fresh DoubleAdapter
        adapter = DoubleAdapter(model.vilt.config.hidden_size, bottleneck)
        layer.output.add_module("adapter", adapter)

        # capture per-layer values in defaults
        orig = layer.output._orig_forward
        this_adapter = adapter

        def patched_forward(hidden_states, input_tensor, _orig=orig, _adapter=this_adapter):
            out = _orig(hidden_states, input_tensor)
            return input_tensor + _adapter(out)

        layer.output.forward = patched_forward

def load_adapters_for_task(model: ViltWithCustomClassifier, adapter_states):
    for layer, state_dict in zip(model.vilt.encoder.layer, adapter_states):
        # make sure we’re on the same device
        state_dict = {k: v.to(device) for k,v in state_dict.items()}
        layer.output.adapter.load_state_dict(state_dict)