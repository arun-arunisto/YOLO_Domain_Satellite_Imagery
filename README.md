# **YOLO Domain Satellite Imagery**

<p align="center">
  <img src="https://img.shields.io/badge/YOLO-v9%20%7C%20v10-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.10+-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Ultralytics-Latest-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" />
</p>

---

This repository contains the full training pipeline for building a **high-quality YOLO-based object detection model** using the **xView Satellite Imagery Dataset**.

The objective is simple:
Build a **strong domain-specific model** optimized for satellite/aerial imagery under the YOLO Domain Hub initiative.

---

## 🚀 Overview

Satellite imagery has its own set of challenges:

* Very small, dense objects
* Huge scale variation
* Rotated and oblique structures
* Heavily cluttered backgrounds
* Large image resolutions

This project is dedicated to producing a **robust YOLO model** tailored specifically for these challenges.

---

## 📦 Dataset: xView

The xView dataset includes:

* **1,000,000+ object annotations**
* **60 classes**
* **0.3 m resolution** satellite scenes
* Urban, rural, maritime, industrial environments

Download the following files manually from the xView dataset page:

```
training_images
validation_images
training_labels (GeoJSON)
```

Dataset link: [https://challenge.xviewdataset.org](https://challenge.xviewdataset.org)

---

## 🔧 Label Conversion: GeoJSON → YOLO

xView labels are polygonal GeoJSON annotations, which are **not YOLO-ready**.

This repo includes a conversion script that:

* Parses polygon coordinates
* Converts polygons → bounding boxes
* Normalizes coordinates
* Generates YOLO-formatted `.txt` labels

Expected YOLO-ready structure:

```
dataset/
 ├── images/
 │     ├── train/
 │     └── val/
 ├── labels/
 │     ├── train/
 │     └── val/
 └── data.yaml
```

---

## 🏗️ Training Pipeline

Start with a basic model:

```bash
yolo detect train \
  model=yolov9s.pt \
  data=data.yaml \
  epochs=100 \
  imgsz=640
```

Then iterate using:

### ✔ YOLO versions

* YOLOv8
* YOLOv9
* YOLOv10

### ✔ Model sizes

(n, s, m, l, x)

### ✔ Hyperparameters

* Learning rate
* Batch size
* Augmentations
* Epochs
* Image size

The purpose is to push for the **best-performing satellite imagery model**.

---

## 📊 Performance Evaluation

Evaluate your trained model:

```bash
yolo detect val \
  model=best.pt \
  data=data.yaml \
  imgsz=640
```

Record the following metrics:

* **mAP50–95**
* **mAP50**
* **Precision**
* **Recall**
* **Per-class performance**
* **Model size + version used**
* **Training parameters**

This ensures reproducible benchmarking.

---

## 📂 Repository Structure

```
YOLO Domain Satellite Imagery/
├── data/                # data.yaml and class names
├── scripts/             # conversion and utility scripts
├── dataset/             # YOLO-ready dataset after conversion
├── models/              # trained model weights/checkpoints
├── notebooks/           # experiments and evaluation notebooks
└── README.md
```

---

## ⚙️ Environment Setup

Install dependencies:

```bash
pip install ultralytics
pip install pillow
pip install shapely
pip install numpy
```

Optional but recommended:

```bash
pip install matplotlib tqdm
```

---

## 📜 License

This project uses:

* **MIT License** for code
* **xView dataset license** (dataset usage restrictions apply)
* **Ultralytics YOLO license** (for training framework)

This repository is intended for open-source research and model development.

---

