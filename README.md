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

## 🛠️ Structure

- `train.py`: Federated training pipeline (multi-user, multi-task)
- `simulator.py`: Network and energy/latency simulator
- `model/`: Adapter-based ViLT model architecture
- `results/`: Logs, accuracy/loss metrics, energy traces
- `utils/`: Data loaders, clustering logic, and helper functions

## 📦 Installation

```bash
git clone https://github.com/payamsiabd/Hierarchical-Multi-Modal-Multi-Task-Federated-Foundation-Model.git
cd Hierarchical-Multi-Modal-Multi-Task-Federated-Foundation-Model
pip install -r requirements.txt
```

## 🚀 Run Training

```bash
python train.py --config configs/hf_fm.yaml
```

You can also change adapter mode, task type, or aggregation settings via the configuration file.

## 📊 Results

All accuracy/loss/energy/latency traces are stored in `results/`. Plots and comparisons are automatically generated for:

- Accuracy vs. Epoch
- Loss vs. Energy/Latency
- System-wide performance under different aggregation schemes

## 📄 License

This project is released under the MIT License.

## 👤 Author

**Payam Abdisarabshali** – Ph.D. Student, Electrical Engineering, SUNY Buffalo

For questions or collaborations, feel free to contact via GitHub or [Google Scholar](https://scholar.google.com/citations?user=ksQpR00AAAAJ&hl=en).

---

Enjoy exploring the future of federated foundation models! 🚀
