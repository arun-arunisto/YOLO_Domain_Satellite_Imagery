---

# **YOLO Domain – Satellite Imagery (DOTA)**

<p align="center">
  <img src="https://img.shields.io/badge/YOLO-v8%20%7C%20v9%20%7C%20v10-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.10+-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Dataset-DOTA-red?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" />
</p>

---

This repository provides a **complete, end-to-end preprocessing and training pipeline** for building **YOLO-based object detection models** on **high-resolution satellite imagery**, using the **DOTA (Dataset for Object Detection in Aerial Images)** dataset.

The objective is simple and strict:

> **Build a clean, reproducible, domain-specific YOLO model for satellite / aerial imagery.**

This work is part of the **YOLO Domain Hub initiative**, focusing on **correct geometry, scalable preprocessing, and reliable benchmarking**.

---

## 🚀 Overview

Satellite and aerial imagery introduce challenges that generic datasets do not:

* Extremely large images (3k–6k resolution)
* Tiny, dense objects
* Large-scale variation
* Heavy background clutter
* Oriented objects (OBB annotations)

This repository addresses these challenges by enforcing a **strict preprocessing pipeline** *before* training YOLO models.

---

## 📦 Dataset: DOTA (v1.0)

**DOTA (Dataset for Object Detection in Aerial Images)** provides:

* High-resolution aerial images
* Oriented bounding box (OBB) annotations
* 15 object categories
* Diverse scenes (urban, ports, airports, industrial zones)

Official dataset page:

```
https://captain-whu.github.io/DOTA/
```

This repository currently supports **DOTA v1.0**.

---

## 🔁 Data Processing Pipeline

DOTA annotations are **not YOLO-compatible**.
This repository converts them through a **multi-stage, geometry-safe pipeline**.

---

### 1️⃣ DOTA → YOLO Label Conversion

* Converts OBB (4-point polygons) → HBB (horizontal bounding boxes)
* Preserves floating-point precision
* Drops difficult objects (configurable)
* Outputs standard YOLO format

YOLO label format:

```
<class_id> <x_center> <y_center> <width> <height>
```

---

### 2️⃣ Image Tiling (Mandatory)

DOTA images are extremely large and **cannot be trained directly**.

Tiling parameters:

* Tile size: **1024 × 1024**
* Overlap: **200 px**
* Bounding boxes are clipped and adjusted per tile
* Very small boxes are filtered to reduce noise

This step is **non-optional** for DOTA-scale imagery.

---

### 3️⃣ Train / Validation Split

* Split is performed **after tiling**
* Deterministic (seeded)
* Ensures image–label alignment
* Produces YOLO-compatible directory layout

Final dataset structure:

```
dataset/
└── tiles_split/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

---

### 4️⃣ Visual Sanity Checking

Before training, labels are **visually inspected**:

* Bounding boxes are drawn on images
* Checked before and after tiling
* Prevents silent geometry errors

This step ensures **training correctness before GPU time is spent**.

---

## 🧠 Processing Philosophy (Important)

* ❌ No resizing full images before tiling

* ❌ No mixing DOTA and YOLO labels

* ❌ No training without visual checks

* ✅ Convert → tile → split → verify → train

* ✅ Skip invalid data aggressively

* ✅ Protect valid annotations

---

## 🏗️ Training Pipeline

Baseline YOLO training example:

```bash
yolo detect train \
  model=yolov8s.pt \
  data=data.yaml \
  imgsz=640 \
  epochs=100
```

Experiments are conducted across:

### ✔ YOLO Versions

* YOLOv8
* YOLOv9
* YOLOv10

### ✔ Model Sizes

* n / s / m / l / x

### ✔ Training Parameters

* Image size
* Batch size
* Epochs
* Augmentations

The goal is **clean benchmarking**, not leaderboard chasing.

---

## 📊 Evaluation Metrics

For every trained model, the following are recorded:

* **mAP50–95** (primary metric)
* **mAP50**
* **Precision**
* **Recall**
* **Per-class performance**
* **Model size & YOLO version**
* **Training configuration**

This ensures **reproducible and comparable results**.

---

## 📂 Repository Structure

```
YOLO Domain Satellite Imagery/
├── data/                     # dataset.yaml, class mappings
├── dataset/                  # processed YOLO-ready dataset
│
├── scripts/
│   ├── dota/                 # Core DOTA processing modules
│   │   ├── __init__.py
│   │   ├── classes.py        # DOTA class definitions
│   │   ├── converter.py      # DOTA → YOLO conversion logic
│   │   ├── datastats.py      # Dataset statistics & analysis
│   │   ├── tiler.py          # Image tiling logic
│   │   └── visualizer.py     # Visual sanity checker
│   │
│   ├── converter_dota.py     # Conversion runner
│   ├── datastats_dota.py     # Statistics runner
│   ├── tiler_dota.py         # Tiling runner
│   └── visualizer_dota.py    # Visualization runner
│
├── models/                   # Trained model weights
├── notebooks/                # Experiments & analysis
└── README.md
```

---

## ⚙️ Environment Setup

Core dependencies:

```bash
pip install ultralytics
pip install numpy pillow opencv-python
```

Recommended utilities:

```bash
pip install matplotlib tqdm
```

---

## 📜 License

* **Code**: MIT License
* **Dataset**: DOTA License (dataset usage restrictions apply)
* **Training Framework**: Ultralytics YOLO License

This repository is intended for **open-source research and reproducible model development**.

---

## 🧭 Project Status

* ✅ DOTA → YOLO conversion
* ✅ Image tiling
* ✅ Train/val split
* ✅ Visual sanity checks
* ⏳ Baseline YOLO training
* ⏳ Benchmarking & reporting

---

## 🧩 YOLO Domain Hub Alignment

This repository is designed to integrate cleanly into the **YOLO Domain Hub**:

* Clear dataset preprocessing
* Reproducible metrics
* Transparent training setup
* Domain-specific focus

---
