import os
import json
from PIL import Image
import torch
from transformers import ViltProcessor
import random
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class NonIIDPartition:
    """Class to handle non-IID partitioning of datasets for federated learning."""

    def __init__(self, dataset, test_data, num_clients_per_cluster, num_clusters, tid, alpha, processor,batch_size=8):
        self.processor = processor
        self.tid = tid
        self.dataset = dataset
        self.clients_per_cluster = []
        self.num_clients = sum(num_clients_per_cluster)
        j = 0
        for i in num_clients_per_cluster:
            self.clients_per_cluster.append([c for c in range(j, j + i)])
            j += i
        self.num_clusters = num_clusters

        self.alpha = alpha
        self.batch_size = batch_size
        self.client_loaders = []
        self.client_train_loaders = []
        self.client_test_loaders = []
        # Global test data loader
        self.test_loader = DataLoader(test_data, batch_size=500, shuffle=False)
        self.client_class_counts = None
        filepath = f'targets{self.tid}.npy'
        if os.path.exists(filepath):
            self.targets = np.load(filepath)
        else:
            self.targets = np.array([dataset[i]['label'] for i in range(len(dataset))])
            np.save(filepath,self.targets)

        self.num_classes = len(np.unique(self.targets))
        self.labels_per_cluster = [self.num_classes]

        assert sum(
            self.labels_per_cluster) >= self.num_classes, "Sum of labels per cluster must be greater or equal to the number of classes"

        self.partition_dataset()


    def partition_dataset(self):

        cluster_labels = []
        label_clients = [[] for i in range(self.num_classes)]
        label_pool = list(range(self.num_classes))
        selected_labels = {c: [] for c in range(len(self.labels_per_cluster))}
        # Assign each label to at least one cluster
        label_pool2 = list(range(self.num_classes))
        np.random.shuffle(label_pool2)
        i = 0
        while len(label_pool2) > 0:
            cluster = np.random.choice(range(len(self.labels_per_cluster)))
            if len(selected_labels[cluster]) < self.labels_per_cluster[cluster]:
                selected_labels[cluster].append(i)
                label_clients[i].extend(self.clients_per_cluster[cluster])
                label_pool2.remove(i)
                i += 1

        # Assign unique labels to each cluster
        c = 0
        for m in self.labels_per_cluster:
            while len(selected_labels[c]) < m:
                label = np.random.choice(label_pool)
                if label not in selected_labels[c]:
                    selected_labels[c].append(label)
                    label_clients[label].extend(self.clients_per_cluster[c])
            cluster_labels.append(selected_labels[c])
            c += 1

        # Dirichlet distribution for partitioning
        indices_per_client = [[] for _ in range(self.num_clients)]
        self.client_class_counts = np.zeros((self.num_clients, self.num_classes))
        for k in range(self.num_classes):
            class_splits = self._dirichlet_partitioner(k, label_clients[k])
            for i, j in zip(label_clients[k], range(len(class_splits))):
                indices_per_client[i].extend(class_splits[j].tolist())
                self.client_class_counts[i, k] = len(class_splits[j])
        if torch.cuda.is_available():
            self.client_loaders = [
                DataLoader(Subset(self.dataset, client_indices), batch_size=self.batch_size, shuffle=True, num_workers =2,  collate_fn=lambda b, proc=self.processor : collate_fn(b, proc))
                    for client_indices in indices_per_client]
        else:
            self.client_loaders = [
                DataLoader(Subset(self.dataset, client_indices), batch_size=self.batch_size, shuffle=True, collate_fn=lambda b, proc=self.processor: collate_fn(b, proc))
                    for client_indices in indices_per_client]

    def stratified_split_dataloader(self, dataloader, split_ratio=0.9):

        dataset = dataloader.dataset
        dataset_size = len(dataset)

        # Assuming dataset has 'targets' attribute which is a list of labels
        targets = [dataset[i][1] for i in range(dataset_size)]

        train_indices, test_indices = train_test_split(
            np.arange(dataset_size), test_size=1 - split_ratio, stratify=targets)

        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=dataloader.batch_size, shuffle=True,
                                  num_workers=dataloader.num_workers)
        test_loader = DataLoader(test_subset, batch_size=dataloader.batch_size, shuffle=False,
                                 num_workers=dataloader.num_workers)

        return train_loader, test_loader

    def _dirichlet_partitioner(self, k, clients):
        # Get indices of samples belonging to current class
        class_indices = np.where(self.targets == k)[0]
        np.random.shuffle(class_indices)

        # Generate proportions of samples for each client using Dirichlet distribution
        proportions = np.random.dirichlet([self.alpha] * len(clients))
        # Remove the last element to avoid out-of-bounds index when splitting
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

        # Split class indices according to the proportions and assign to clients
        class_splits = np.split(class_indices, proportions)
        return class_splits

    def plot_dirichlet_distribution(self,ax,ll):
        """Plots the Dirichlet distribution of data among clients."""
        # Normalize for plotting
        normalized_counts = self.client_class_counts / self.client_class_counts.sum(axis=1, keepdims=True)

        # Plot the stacked horizontal bar plot with improved aesthetics

        bottom = np.zeros(self.num_clients)
        num_classes = self.client_class_counts.shape[1]
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

        for c in range(num_classes):
            ax.barh(range(self.num_clients), normalized_counts[:, c], left=bottom, color=colors[c], edgecolor='k',
                     label=f'Class {c}')
            bottom += normalized_counts[:, c]

        # Add cluster brackets outside the client numbers
        clients_per_cluster = self.num_clients // self.num_clusters
        for i in range(self.num_clusters):
            start = i * clients_per_cluster
            end = (i + 1) * clients_per_cluster - 1
            mid = (start + end) / 2
            bracket_x = -0.15  # Adjust as needed to place bracket outside of the plot
            ax.annotate('', xy=(bracket_x, start), xytext=(bracket_x, end),
                         arrowprops=dict(arrowstyle=']-[', lw=2, color='black', linestyle='-'), annotation_clip=False)
            ax.text(bracket_x - 0.05, mid, f'User {i + 1}', va='center', ha='right', fontsize=12, weight='bold')

        ax.set_xlabel("Number of labels per user = "+str(ll))
        # ax.set_ylabel('Number of FLUs')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        # ax.tight_layout()


    def get_client_loaders(self):
        """Returns the data loaders for each client."""
        return self.client_loaders

    def get_test_loader(self):
        """Returns the global test data loader."""
        return self.test_loader

    def get_test_loaders(self):
        """Returns the global test data loader."""
        return self.client_test_loaders

    def get_train_loaders(self):
        """Returns the global test data loader."""
        return self.client_train_loaders




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