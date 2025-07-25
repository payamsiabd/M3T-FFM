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
from HFFM.core.datasets import *
import argparse
# from HFFM.core.models import *
# from HFFM.core.utils import *

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocalGQADataset(Dataset):
    def __init__(self, ann_path: str, img_dir: str, label2id: dict):
        self.img_dir = img_dir
        self.label2id = label2id
        with open(ann_path, "r") as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(os.path.join(self.img_dir, rec["image_id"])).convert("RGB")
        question = rec["question"]
        answer = rec["answer"]
        label_id = self.label2id.get(answer, -100)
        return {"image": image, "question": question, "label": label_id}

class LocalArtDataset(Dataset):
    def __init__(self, ann_path: str, img_dir: str, label2id: dict):
        self.img_dir   = img_dir
        self.label2id  = label2id
        with open(ann_path, "r") as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image    = Image.open(os.path.join(self.img_dir, rec["image"])).convert("RGB")
        question = rec["question"]
        answer   = rec["answer"]
        label_id = self.label2id.get(answer, -100)
        return {"image": image, "question": question, "label": label_id}

class LocalVizWizDataset(Dataset):
    def __init__(self, ann_path: str, img_dir: str, label2id: dict):
        self.img_dir = img_dir
        self.label2id = label2id
        with open(ann_path, 'r') as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Load image
        img_path = os.path.join(self.img_dir, rec["image"])
        image = Image.open(img_path).convert("RGB")

        # Get the question
        question = rec["question"]

        # Collect all answer strings
        answers = [a["answer"].strip().lower() for a in rec["answers"] if a.get("answer")]

        # Use the most common answer, if it's in label2id
        label = -100
        if answers:
            counter = Counter(answers)
            most_common_answer, _ = counter.most_common(1)[0]
            if most_common_answer in self.label2id:
                label = self.label2id[most_common_answer]
            else:
                # Fallback: use the first answer in the list that is in label2id
                for a in answers:
                    if a in self.label2id:
                        label = self.label2id[a]
                        break

        return {"image": image, "question": question, "label": label}


def load_mappings(in_dir="HFFM/datasets/vizwiz"):
    # vocab (not strictly needed here, but returned for completeness)
    with open(os.path.join(in_dir, "vocab.txt"), "r") as f:
        vocab = [line.strip() for line in f if line.strip()]
    with open(os.path.join(in_dir, "label2id.json"), "r") as f:
        label2id = json.load(f)
    with open(os.path.join(in_dir, "id2label.json"), "r") as f:
        raw = json.load(f)
        # JSON keys are strings, convert back to int
        id2label = {int(k):v for k,v in raw.items()}
    return vocab, label2id, id2label

def collate_fn(batch, processor: ViltProcessor):
    images   = [ex["image"]   for ex in batch]
    questions= [ex["question"]for ex in batch]
    labels   = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)

    inputs = processor( images=images, text=questions, padding="max_length", truncation=True, max_length=processor.tokenizer.model_max_length, return_tensors="pt")
    inputs["labels"] = labels
    return inputs


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

        self.classifier = [None,None]
        self.tid = -1

    def set_classifier(self, classifier: nn.Module, tid):
        """Attach one of your external heads to this model."""
        self.classifier[tid] = classifier
        self.classifier[tid].to(device)
        # make sure its parameters will be trained
        for p in self.classifier[tid].parameters():
            p.requires_grad = True
            
    def forward_task(self, tid,**inputs):
        self.tid = tid
        return self.forward(**inputs)

    def forward(self,**inputs):
        assert self.classifier[self.tid] is not None
        outputs = self.vilt(**inputs)
        pooled_output = outputs.pooler_output
        logits = self.classifier[self.tid](pooled_output)
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


def average_distance(param_g, param):
    total_distance = 0.0
    num_params = 0

    for pg, p in zip(param_g, param):
        if pg.requires_grad and p.requires_grad:
            distance = torch.norm(pg.data - p.data, p=2)  # L2 norm
            total_distance += distance.item()
            num_params += 1

    return total_distance / num_params if num_params > 0 else 0.0


def flatten_params(param_dict, prefix):
    parts = []
    for k, v in sorted(param_dict.items()):
        if k.startswith(prefix):
            parts.append(v.view(-1))
    if len(parts)==0:
        return None
    return torch.cat(parts, dim=0)

def head_vector(head_module):
    sd = head_module.state_dict()
    return torch.cat([sd["weight"].view(-1), sd["bias"].view(-1)], dim=0)

def compute_all_similarities(personalized_adapters, classifier_modules, task_ids):
    tid1, tid2 = task_ids
    num_layers = len(personalized_adapters[tid1])
    sims = {"ad1":[], "ad2":[]}

    # adapters:
    for layer_idx in range(num_layers):
        state1 = personalized_adapters[tid1][layer_idx]
        state2 = personalized_adapters[tid2][layer_idx]
        for comp in ["ad1.", "ad2."]:
            v1 = flatten_params(state1, comp)
            v2 = flatten_params(state2, comp)
            if v1 is not None and v2 is not None:
                sims[comp.rstrip(".")].append(
                    F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item()
                )
    # heads:
    h1 = head_vector(classifier_modules[tid1])
    h2 = head_vector(classifier_modules[tid2])
    head_sim = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0), dim=1).item()

    # average across layers:
    return {
      "avg_ad1_sim": sum(sims["ad1"]) / len(sims["ad1"]),
      "avg_ad2_sim": sum(sims["ad2"]) / len(sims["ad2"]),
      "head_sim": head_sim
    }
def evaluate(model, test_loader, task_id, ep, is_test):
    model.eval()
    correct, total = 0, 0
    all_true = []
    all_pred = []
    with torch.no_grad():
        # loop = tqdm(test_loader, desc=f"Evaluating...", unit="batch")
        if is_test:
            loop = tqdm(test_loader, desc=f"Evaluating...", unit="batch")
        else:
            loop = test_loader
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            logits = model.forward_task(task_id,**batch)
            preds  = logits.argmax(dim=-1)

            mask   = labels != -100
            correct += (preds[mask] == labels[mask]).sum().item()
            total   += mask.sum().item()
            true   = labels[mask].cpu().tolist()
            pred   = preds[mask].cpu().tolist()

            all_true += true
            all_pred += pred

    acc = 100 * correct / total if total>0 else 0.0
    print(f"[Epoch {ep}], Task {task_id}, test accuracy = {acc:.2f}%\n")
    return acc

def save_model_components(classifier_modules, personalized_adapters2, personalized_adapters, save_dir="local_saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    # Save classifier modules
    for (tid, uid), clf in classifier_modules.items():
        torch.save(clf.state_dict(), os.path.join(save_dir, f"classifier_tid{tid}_uid{uid}.pt"))

    # Save personalized_adapters2
    for uid, layers in personalized_adapters2.items():
        torch.save(layers, os.path.join(save_dir, f"personalized_adapter2_uid{uid}.pt"))

    # Save personalized_adapters
    for tid, user_adapters in personalized_adapters.items():
        for uid, layers in enumerate(user_adapters):
            torch.save(layers, os.path.join(save_dir, f"personalized_adapter_tid{tid}_uid{uid}.pt"))

def load_model_components(num_labels, n_users, task_ids, model_class, hidden_size, save_dir="saved_models"):
    classifier_modules = {}
    for tid, nl in num_labels.items():
        for uid in range(n_users):
            clf = model_class(hidden_size, nl)
            path = os.path.join(save_dir, f"classifier_tid{tid}_uid{uid}.pt")
            clf.load_state_dict(torch.load(path))
            classifier_modules[(tid, uid)] = clf

    personalized_adapters2 = {}
    for uid in range(n_users):
        path = os.path.join(save_dir, f"personalized_adapter2_uid{uid}.pt")
        personalized_adapters2[uid] = torch.load(path)

    personalized_adapters = {}
    for tid in task_ids:
        personalized_adapters[tid] = []
        for uid in range(n_users):
            path = os.path.join(save_dir, f"personalized_adapter_tid{tid}_uid{uid}.pt")
            personalized_adapters[tid].append(torch.load(path))

    return classifier_modules, personalized_adapters2, personalized_adapters



def save_model_components(classifier_modules, personalized_adapters, personalized_adapters2, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    # Save classifier modules
    for (tid, uid), clf in classifier_modules.items():
        torch.save(clf.state_dict(), os.path.join(save_dir, f"classifier_tid{tid}_uid{uid}.pt"))

    # Save personalized_adapters (by cluster id)
    for cid, layer_state_dicts in personalized_adapters.items():
        torch.save(layer_state_dicts, os.path.join(save_dir, f"personalized_adapter_cid{cid}.pt"))

    # Save personalized_adapters2 (by task id and user id)
    for tid, user_adapters in personalized_adapters2.items():
        for uid, layer_state_dicts in enumerate(user_adapters):
            torch.save(layer_state_dicts, os.path.join(save_dir, f"personalized_adapter2_tid{tid}_uid{uid}.pt"))


def load_model_components(num_labels, n_users, task_ids, clusters, model_class, hidden_size, save_dir="saved_models"):
    classifier_modules = {}
    for tid, nl in num_labels.items():
        for uid in range(n_users):
            clf = model_class(hidden_size, nl)
            path = os.path.join(save_dir, f"classifier_tid{tid}_uid{uid}.pt")
            clf.load_state_dict(torch.load(path))
            classifier_modules[(tid, uid)] = clf

    personalized_adapters = {}
    for cid in clusters.keys():
        path = os.path.join(save_dir, f"personalized_adapter_cid{cid}.pt")
        personalized_adapters[cid] = torch.load(path)

    personalized_adapters2 = {}
    for tid in task_ids:
        personalized_adapters2[tid] = []
        for uid in range(n_users):
            path = os.path.join(save_dir, f"personalized_adapter2_tid{tid}_uid{uid}.pt")
            personalized_adapters2[tid].append(torch.load(path))

    return classifier_modules, personalized_adapters, personalized_adapters2


def main(args):

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    base_dirs = ["HFFM/datasets/gqa","HFFM/datasets/art2"]
    task_ids = [0,1]
    label2ids = {}
    vocabs= {}
    num_labels = {}
    for tid in task_ids:
        vocab, label2id, _ = load_mappings(base_dirs[tid])
        label2ids[tid]=label2id
        vocabs[tid]=vocab
        num_labels[tid]=len(vocab)
        
    # 1) load processor & model
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    base_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltWithCustomClassifier(base_model)
    
    add_adapters_to_vilt(model, bottleneck=256)

    model.to(device)
    datasets = {}
    train_sizes = {}
    test_sizes = {}
    train_loaders = {}
    test_loaders = {}

    num_dirichlet_clusters = args.num_dirichlet_clusters
    n_users               = args.n_users
    alpha                 = args.alpha
    is_test               = False
    epochs                = args.epochs
    cons_rounds           = args.cons_rounds
    users_per_cluster     = args.users_per_cluster
    n_clusters            = int(n_users/users_per_cluster)
    gaf                   = args.gaf
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    accuracy_record = []
    path = f"HFFM/results/hierarchy_{gaf}_{users_per_cluster}"
    os.makedirs(path, exist_ok=True)
    save_dir = f"hierarchy_{gaf}_{users_per_cluster}"

    clients_per_cluster = [n_users]
    partitioner = {}
    user_loaders = {}
    user_test_loaders = {}

    for tid in task_ids:
        annotation_path = os.path.join(base_dirs[tid], "ann.json")
        dataset_path = os.path.join(base_dirs[tid],"train")
        if tid==0:
            datasets[tid]=LocalGQADataset(annotation_path,dataset_path, label2ids[tid])
        elif tid==1:
            datasets[tid]=LocalArtDataset(annotation_path,dataset_path, label2ids[tid])
        elif tid==2:
            datasets[tid]=LocalVizWizDataset(annotation_path,dataset_path, label2ids[tid])

        # Split into 90% train / 10% test
        train_sizes[tid]=int(0.9 * len(datasets[tid]))
        test_sizes[tid]=len(datasets[tid]) - train_sizes[tid]
        # train_sizes[tid]=int(0.01 * len(datasets[tid]))
        # test_sizes[tid]=int(0.01 * len(datasets[tid]))
        # dump = len(datasets[tid])-(train_sizes[tid]+test_sizes[tid])

        train_ds, test_ds= torch.utils.data.random_split(datasets[tid], [train_sizes[tid], test_sizes[tid]], generator=torch.Generator().manual_seed(SEED))

        partitioner[tid] = NonIIDPartition(dataset=train_ds, test_data=test_ds,num_clients_per_cluster=clients_per_cluster, tid = tid,
                                      num_clusters=num_dirichlet_clusters, alpha=alpha, processor=processor, batch_size=8)
        user_loaders[tid] = partitioner[tid].client_loaders

        # tot = len(train_ds)
        # base = tot // n_users
        # sizes = [base + (1 if i<tot % n_users else 0) for i in range(n_users)]
        # shards = torch.utils.data.random_split(train_ds, sizes, generator=torch.Generator().manual_seed(SEED))
        # # wrap each shard in a DataLoader
        # loaders = [ DataLoader(shard, batch_size=8, shuffle=True, num_workers=4, collate_fn=lambda b, proc=processor: collate_fn(b, proc)) for shard in shards]
        # user_loaders[tid]=loaders


        test_loaders[tid]=DataLoader(test_ds, batch_size=8, shuffle=False,  num_workers =2, collate_fn=lambda b: collate_fn(b, processor))

        base = len(test_ds) // n_users
        sizes = [base + (1 if i<len(test_ds) % n_users else 0) for i in range(n_users)]
        shards = torch.utils.data.random_split(test_ds, sizes, generator=torch.Generator().manual_seed(SEED))
        user_test_loaders[tid]=[ DataLoader(shard, batch_size=8, shuffle=True, num_workers=2, collate_fn=lambda b, proc=processor: collate_fn(b, proc)) for shard in shards]

    # 6) optimizer & loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # clusters = np.load("HFFM/clusters.npy", allow_pickle=True).item()

    clusters = {cid: [users_per_cluster * cid + i for i in range(users_per_cluster)] for cid in range(n_clusters)}
    print(clusters)

    classifier_modules = {}
    for tid, nl in num_labels.items():
        main_head =nn.Linear(model.hidden_size, nl)
        for uid in range(n_users):
            classifier_modules[(tid, uid)]=copy.deepcopy(main_head)
        
    personalized_adapters = {cid: [{k: v.detach().cpu().clone() for k, v in layer.output.adapter.state_dict().items()} for layer in model.vilt.encoder.layer] for cid in clusters.keys()}
    personalized_adapters2 = {tid: [[{ k: v.detach().cpu().clone() for k, v in layer.output.adapter.state_dict().items()} for layer in model.vilt.encoder.layer ] for uid in range(n_users)] for tid in task_ids}

    for ep in range(1, epochs+1):
        print(f"\n--- Global Epoch {ep} ---")
        avg_acc = [0,0]

        adapter_accum_global = {k: torch.zeros_like(v, device="cpu") for k, v in model.state_dict().items() if "adapter" in k}
        adapter_count_global = 0
        # 1b) Head accumulators (one per task)
        head_accum_global  = {tid: { k: torch.zeros_like(v, device="cpu") for k, v in classifier_modules[(tid,0)].state_dict().items() } for tid in task_ids}
        head_count_global = {tid: 0 for tid in task_ids}

        for cid, users in clusters.items():
            adapter_accum = {k: torch.zeros_like(v, device="cpu") for k, v in model.state_dict().items() if "adapter" in k}
            adapter_count = 0
            # 1b) Head accumulators (one per task)
            head_accum  = {tid: { k: torch.zeros_like(v, device="cpu") for k, v in classifier_modules[(tid,0)].state_dict().items() } for tid in task_ids}
            head_count = {tid: 0 for tid in task_ids}
            for uid in users:
                model.classifier=[None, None]
                local_model = model
                local_adapters =[{ k: v.clone() for k, v in layer_state.items() } for layer_state in personalized_adapters[cid]]
                load_adapters_for_task(local_model,local_adapters)

                for tid in task_ids:
                    
                    head = classifier_modules[(tid, uid)]
                    local_model.set_classifier(head, tid)

                    trainable = list(local_model.classifier[tid].parameters()) + [p for n,p in local_model.vilt.named_parameters() if "adapter" in n]
                    optim = AdamW(trainable, lr=1e-4)
                    total_loss = 0.0
                    local_model.train()
                    # loop = tqdm(user_loaders[tid][uid], desc=f"Epoch {ep}, Task {tid}, User {uid} ", unit="batch")
                    N_i = len(user_loaders[tid][uid].dataset)
                    if is_test:
                        loop = tqdm(user_loaders[tid][uid], desc=f"Epoch {ep}, Task {tid}, User {uid} ", unit="batch")
                    else:
                        loop = user_loaders[tid][uid]
                    total_examples = 0
                    for batch in loop:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        labels = batch.pop("labels")
                        outputs = local_model.forward_task(tid,**batch)
                        logits  = outputs
                        loss    = criterion(logits, labels)
                        loss.backward()
                        optim.step()
                        optim.zero_grad()
                        total_loss += loss.item()
                    
                    print(f"{total_loss}:{len(user_loaders[tid][uid])}")
                    print(f"[Epoch {ep}], Cluster {cid}, Task {tid}, User {uid}, train loss = {total_loss/len(user_loaders[tid][uid]):.4f}")
                    acc = evaluate(local_model, user_test_loaders[tid][uid],tid, ep, is_test)
                    avg_acc[tid]+=acc

                    accuracy_record.append({
                                "epoch":    ep,
                                "cluster":  cid,
                                "user":     uid,
                                "task":     tid,
                                "loss":     total_loss/len(user_loaders[tid][uid]),
                                "accuracy": acc
                            })
                
                    with open(f"{path}/accuracy_record.json", "w") as f:
                        json.dump(accuracy_record, f, indent=4)

                    sd = local_model.state_dict()

                    # a) adapters
                    for k in adapter_accum:
                        adapter_accum[k] += sd[k].detach().cpu()*N_i
                    adapter_count += N_i
                    

                    # b) this task’s head
                    hsd = head.state_dict()
                    for k in head_accum[tid]:
                        head_accum[tid][k] += hsd[k].detach().cpu()*N_i
                    head_count[tid] += N_i

                    if ep%gaf == 0:
                        for k in adapter_accum_global:
                            adapter_accum_global[k] += adapter_accum[k]
                        adapter_count_global += adapter_count
                        
                        for k in head_accum_global[tid]:
                            head_accum_global[tid][k] += head_accum[tid][k]
                        head_count_global[tid] += head_count[tid]


                    del optim      
                    gc.collect()              
                    torch.cuda.empty_cache()
            print(f"Local aggregation of cluster {cid}...")
            for tid in task_ids:
                # Personalized adapter aggregation
                for layer_idx, layer_state in enumerate(personalized_adapters[cid]):
                    new_layer_state = {}
                    for param_name in layer_state.keys():
                        full_key = f"vilt.encoder.layer.{layer_idx}.output.adapter.{param_name}"
                        avg = adapter_accum[full_key] / adapter_count
                        new_layer_state[param_name] = avg.clone()
                    
                    personalized_adapters[cid][layer_idx] = new_layer_state

                # Classification head
                cnt = head_count[tid]
                new_head_sd = {}
                for k, tot in head_accum[tid].items():
                    new_head_sd[k] = (tot / cnt).to(device)
                for uid in users:
                    classifier_modules[(tid,uid)].load_state_dict(new_head_sd)

        for tid in task_ids:
            print(f"[Epoch {ep}], avg test accuracy of task {tid} = {avg_acc[tid]/n_users:.2f}%\n")
        
        if ep%gaf == 0:
            print(f"Global aggregation...")
            for tid in task_ids:
                # Personalized adapter aggregation
                for layer_idx, layer_state in enumerate(personalized_adapters[0]):
                    new_layer_state = {}
                    for param_name in layer_state.keys():
                        full_key = f"vilt.encoder.layer.{layer_idx}.output.adapter.{param_name}"
                        avg = adapter_accum_global[full_key] / adapter_count_global
                        new_layer_state[param_name] = avg.clone()
                    for cid in clusters.keys():
                        personalized_adapters[cid][layer_idx] = new_layer_state

                # Classification head
                cnt = head_count_global[tid]
                new_head_sd = {}
                for k, tot in head_accum_global[tid].items():
                    new_head_sd[k] = (tot / cnt).to(device)
                for cid, users in clusters.items():
                    for uid in users:
                        classifier_modules[(tid,uid)].load_state_dict(new_head_sd)
        
        save_model_components(classifier_modules, personalized_adapters, personalized_adapters2, save_dir = save_dir)



def execute():
    parser = argparse.ArgumentParser(description="Run federated VILT with adapters")
    parser.add_argument("--num_dirichlet_clusters", type=int,   default=1)
    parser.add_argument("--n_users",               type=int,   default=40)
    parser.add_argument("--alpha",                 type=float, default=0.1)
    parser.add_argument("--is_test",               action="store_true")
    parser.add_argument("--epochs",                type=int,   default=200)
    parser.add_argument("--cons_rounds",           type=int,   default=100)
    parser.add_argument("--n_clusters",            type=int,   default=10)
    parser.add_argument("--users_per_cluster",     type=int,   default=4)
    parser.add_argument("--gaf",                   type=int,   default=2)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    execute()
