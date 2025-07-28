# Hierarchical Multi-Modal Multi-Task Federated Foundation Model (HF-FM)

This repository contains the official implementation for **Hierarchical Multi-Modal Multi-Task Federated Foundation Model (HF-FM)**, a system that integrates vision-language foundation models with hierarchical federated learning (FL) to enable efficient, scalable, and personalized training across heterogeneous edge networks.

## 🔍 Overview

HF-FM is designed to support diverse **modalities** (e.g., vision, language), multiple **tasks** (e.g., classification, grounding, captioning), and federated **hierarchies** (users → edge → cloud). It uses modular adapters and task heads to support dynamic task execution and personalization across users, edge nodes, and cloud servers.

Key features:

- Multi-modal support via pretrained ViLT backbone
- Multi-task heads (classification, grounding, etc.)
- Adapter-based personalization with low memory usage
- Hierarchical FL orchestration with device-to-device (D2D) aggregation
- Energy- and latency-aware simulation engine

## 🌐 Network Model

The HF-FM system's network model is inspired by and extends the following works:

- (https://ieeexplore.ieee.org/abstract/document/9705093): Multi-Stage Hybrid Federated Learning Over Large-Scale D2D-Enabled Fog Networks
- (https://ieeexplore.ieee.org/document/10304380): Delay-Aware Hierarchical Federated Learning
- (https://arxiv.org/abs/2404.06324): Dynamic D2D-Assisted Federated Learning over O-RAN: Performance Analysis, MAC Scheduler, and Asymmetric User Selection
- (https://ieeexplore.ieee.org/document/9148862): Client-Edge-Cloud Hierarchical Federated Learning

## 📁 Project Structure

```
HFFM/
├── core/
│   ├── datasets.py              # Dataset loading and preprocessing
│   ├── models.py                # Adapter-based model definitions
│   ├── network.py               # Communication & aggregation logic
│   └── utils.py                 # Helper functions
│
├── datasets/
│   ├── dataset_generator_*.py   # Scripts to generate balanced datasets
│   ├── *_vocab_balanced.py      # Balanced vocab files for tasks (ART, GQA, VizWiz)
│
├── methods/
│   ├── main_hierarchy.py        # Entry point for hierarchical FL
│   ├── main_local.py            # Entry point for local-only training
│   ├── table_generator.py       # Summarize evaluation metrics
│   └── results/                 # Folder to store results and logs
```

## 📦 Installation

```bash
git clone https://github.com/payamsiabd/M3T-FFM.git
```

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure PyTorch with GPU support is installed and datasets are available locally.

### 2. Prepare Datasets
The experiments in this project are conducted on two Visual Question Answering (VQA) datasets:

- **ArtVQA**: [ArtVQA (AQUA subset)](https://github.com/noagarcia/ArtVQA/tree/master/AQUA)
- **GQA**: [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html)
- **Anotation files**: [Anotations](https://drive.google.com/drive/folders/1wlx22Y8KGPmrRFELj2su8CSU5dM0Iu_1?usp=drive_link)
- 
Preprocess datasets:

```bash
python datasets/dataset_generator_gqa_balanced.py
python datasets/dataset_generator_art_balanced.py
python datasets/gqa_vocab_balanced.py
python datasets/art_vocab_balanced.py
```

### 3. Run Training

#### Local FL (no aggregation):
```bash
python methods/main_local.py
```

#### Hierarchical FL (e.g., HF-FM):
```bash
python methods/main_hierarchy.py
```

## 📊 Results

Below is the summary of performance on multi-modal multi-task datasets using various aggregation strategies.

| **Method**                              | **Latency (s)** | **Energy (kJ)** | **Test Accuracy (%)** |
|----------------------------------------|-----------------|-----------------|------------------------|
| FFM with Conventional Star Topology    | 12.10           | 387.20          | 37.89                  |
| FFM with Only Edge Aggregation         | 3.59            | 59.25           | 27.48 †                |
| FFM with Only Edge Aggregation + D2D   | 2.00            | 14.75           | 27.48 †                |
| `HF-FM` (E<sub>Agg</sub>=2)            | 9.58            | 107.65          | **38.65** ★            |
| `HF-FM` (E<sub>Agg</sub>=8)            | 5.04            | 71.35           | 36.25 ◁                |
| `HF-FM + D2D` (E<sub>Agg</sub>=2)      | 8.86            | 67.57           | **38.65** ★            |
| `HF-FM + D2D` (E<sub>Agg</sub>=8)      | 3.66            | 27.92           | 36.25 ◁                |

†, ◁, ★ denote statistically indistinguishable results in accuracy.

## 📄 License

This project is released under the MIT License.

## 👤 Author

**Payam Abdisarabshali** – Ph.D. Student, Electrical Engineering, The State University of New York at Buffalo

For questions or collaborations, feel free to contact via GitHub or [Google Scholar](https://scholar.google.com/citations?user=ksQpR00AAAAJ&hl=en).

