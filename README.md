# 👥 Crowd Counting via Density Maps & CSRNet

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-ShanghaiTech_Part_A_%26_B-blue.svg)](https://svip-lab.github.io/dataset/campus_dataset.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning pipeline for estimating crowd count and continuous spatial density heatmaps from congested images using **CSRNet (Dilated Convolutions)** and **Geometry-Adaptive Gaussian Filtering** evaluated on the gold-standard **ShanghaiTech Dataset (Part A & B)**.

---

## 📌 Key Highlights
- **Continuous Density Formulation**: Solves extreme occlusion ($>80\%$) where standard object detectors (YOLO, Faster R-CNN) fail.
- **Strict Count Conservation**: Mathematically normalized adaptive Gaussian kernels guarantee:
  $$\text{Crowd Count} = \iint D(x, y) \, dx \, dy \approx \sum_{x, y} D(x, y)$$
- **CSRNet Architecture**: First 10 layers of VGG-16 frontend paired with dilated convolutional backend ($r=2$) to maintain spatial resolution without aggressive pooling.
- **Dual Pipeline**:
  - Full interactive, pre-executed Jupyter Notebook: [`crowd_counting_density_maps.ipynb`](./crowd_counting_density_maps.ipynb).
  - Standalone command-line inference script: [`run_project.py`](./run_project.py).

---

## 🏗️ Architecture Overview

```
Input Image (H x W x 3)
         │
         ▼
[VGG-16 Frontend (Conv1 to Conv4)] ──► Deep Multi-scale Feature Extraction
         │
         ▼
[Dilated Backend (Dilation Rate = 2)] ──► Expands Receptive Field without Downsampling
         │
         ▼
[1x1 Conv Output Layer]
         │
         ▼
Predicted Density Map D(x, y) (H/8 x W/8)
         │
         ├─► Pixel Sum (Integral) ──► Total Crowd Count
         └─► Jet Colormap Blending ──► High-Resolution Congestion Heatmap Overlay
```

---

## 📊 Qualitative Demonstration

The model predicts continuous density distributions and overlays heatmaps over crowded scenes:

| Raw Input Scene | Predicted Density Map | Density Heatmap Overlay |
| :---: | :---: | :---: |
| High-congestion crowd photo | Reconstructed Gaussian density | Blended spatial density heat zones |

*(See `sample_shanghaitech_result.png` for generated visual output)*

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/crowd-counting-density-maps.git
cd crowd-counting-density-maps
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Standalone Script
Run evaluation and inference on ShanghaiTech directly:
```bash
python run_project.py
```

### 4. Interactive Notebook
Open [`crowd_counting_density_maps.ipynb`](./crowd_counting_density_maps.ipynb) in Jupyter Notebook or VS Code to explore the theoretical formulation, data pipeline, and training curves.

---

## 📦 Dataset: ShanghaiTech
This project benchmarks on **ShanghaiTech** (Zhang et al., CVPR 2016):
- **Part A** ($482\text{ images}$): Extremely congested crowds from web photos (Average: $501$ people/image).
- **Part B** ($716\text{ images}$): Urban surveillance feeds from Shanghai streets (Average: $123$ people/image).

To download and extract ShanghaiTech:
```bash
python -m gdown 16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI -O shanghaitech.zip
python -c "import zipfile; zipfile.ZipFile('shanghaitech.zip').extractall('ShanghaiTech')"
```

---

## 📖 Citations & References
- Zhang et al., *"Single-Image Crowd Counting via Multi-Column Convolutional Neural Network"*, CVPR 2016.
- Li et al., *"CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes"*, CVPR 2018.
- Wang et al., *"DM-Count: Distribution Matching for Crowd Counting"*, NeurIPS 2020.
