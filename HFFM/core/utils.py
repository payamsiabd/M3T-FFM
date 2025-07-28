
import os
import torch
import random
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



def save_model_components(classifier_modules, personalized_adapters, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    # Save classifier modules
    for (tid, uid), clf in classifier_modules.items():
        torch.save(clf.state_dict(), os.path.join(save_dir, f"classifier_tid{tid}_uid{uid}.pt"))

    # Save personalized_adapters (by cluster id)
    for cid, layer_state_dicts in personalized_adapters.items():
        torch.save(layer_state_dicts, os.path.join(save_dir, f"personalized_adapter_cid{cid}.pt"))


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


    return classifier_modules, personalized_adapters
