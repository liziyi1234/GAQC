
# GAQC: Efficient Blind Omnidirectional Image Quality Assessment

<p align="center">
  <b>Efficient Blind Omnidirectional Image Quality Assessment: A Two-Dimensional Perspective</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#framework">Framework</a> •
  <a href="#installation">Installation</a> •
  <a href="#data-preparation">Data Preparation</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#performance">Performance</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-BOIQA-blue.svg">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange.svg">
  <img src="https://img.shields.io/badge/Input-ERP-green.svg">
  <img src="https://img.shields.io/badge/Parameters-4.7M-brightgreen.svg">
  <img src="https://img.shields.io/badge/FLOPs-1.5G-yellow.svg">
</p>

---

## 📖 Overview

**GAQC** is an efficient blind omnidirectional image quality assessment framework that directly takes a complete equirectangular projection (ERP) image as input and predicts its perceptual quality score.

Unlike viewport-aware methods that require viewport extraction or viewport quality prediction, GAQC performs quality assessment directly on the complete ERP representation. From a two-dimensional perspective, GAQC formulates blind omnidirectional image quality assessment as a blind image quality assessment problem while explicitly considering the geometric characteristics introduced by the ERP representation.

GAQC is designed with a lightweight architecture and achieves efficient quality prediction with only **4.7M parameters** and **1.5G FLOPs**.

The overall pipeline can be summarized as:

<p align="center">

**ERP Image → Feature Extraction → GA Module → QCR Module → Quality Score**

</p>

---

## ✨ Key Features

### ⚡ Lightweight and Efficient

GAQC is designed for efficient blind omnidirectional image quality assessment.

- Only **4.7M parameters**.
- Approximately **1.5G FLOPs**.
- Directly processes complete ERP images.
- Does not require viewport extraction or viewport prediction.

This lightweight design enables efficient quality prediction while maintaining effective perceptual quality modeling.

---

### 🌐 GA Module: Geometry-Aware Feature Modeling

The **GA module** explicitly models the geometric characteristics of ERP images and adaptively corrects geometry-related feature distortions.

The module integrates:

- **Spherical coordinate encoding** to provide explicit positional information.
- **Frequency attention** to adaptively emphasize informative frequency components.
- **Deformable convolution** to adapt feature sampling according to geometric variations.

By combining these components, the GA module enables adaptive geometric correction and improves feature representations under the non-uniform spatial characteristics of ERP images.

---

### 🔍 QCR Module: Quality Context Representation

The **Quality Context Representation (QCR)** module integrates global contextual information with multi-scale local features for perceptual quality modeling.

- Captures global image-level quality context.
- Extracts complementary local quality information at multiple spatial scales.
- Fuses global and local representations for robust quality prediction.

This design enables GAQC to jointly model both holistic quality perception and local distortion patterns.

---

## 🏗 Framework

The overall architecture of GAQC is illustrated below.

<p align="center">
  <img src="figures/framework.png" width="90%">
</p>

GAQC consists of two key components:

| Module | Full Name | Description |
|:---:|:---|:---|
| **GA** | Geometry-Aware Module | Models spherical geometric characteristics through coordinate encoding, frequency attention, and deformable convolution |
| **QCR** | Quality Context Representation | Integrates global quality context with multi-scale local feature representations |

The two modules collaboratively model geometry-aware and quality-relevant representations for efficient blind omnidirectional image quality assessment.

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/liziyi1234/GAQC.git
cd GAQC
```

---

##  📂 Data Preparation

Before training, please prepare the corresponding omnidirectional image quality assessment dataset and configure the dataset paths according to your local environment.

A typical dataset organization is shown below:

```bash
GAQC/
├── datasets/
│   ├── OIQA_Dataset/
│   │   ├── images/
│   │   ├── train.txt
│   │   └── test.txt
│   │
│   └── Another_Dataset/
│       ├── images/
│       ├── train.txt
│       └── test.txt
│
├── GAQC.py
├── train.py
├── config.py
├── MyDataset.py
└── utils.py
```

The image paths, subjective quality scores, and dataset splits should be organized according to the format required by MyDataset.py.

Before running the code, please configure the dataset path and other experimental settings in:

```bash
config.py
```

**Note:** The dataset organization may vary depending on the specific database. Please ensure that the image paths, quality scores, and train/test splits are correctly specified before training.

---


##  🚀 Training

To train GAQC, run:

```bash
python train.py
```

The main training configuration can be adjusted in:

```bash
config.py
```

Depending on your experimental setting, you may modify parameters such as:

```bash
Dataset path
Dataset name
Training and validation splits
Batch size
Learning rate
Number of epochs
Optimizer settings
Random seed
Model-related configurations
```

---

##  🔮 Inference

After training, GAQC can be used to predict the perceptual quality score of an input ERP image.

Please specify the path to the trained model checkpoint and the input image before inference.

---

##  📊 Performance

GAQC is designed to provide an efficient solution for blind omnidirectional image quality assessment.

Unlike viewport-aware approaches that process multiple viewports, GAQC directly performs quality assessment on the complete ERP image. This design avoids the additional computational cost introduced by viewport extraction and prediction.

The effectiveness of GAQC is mainly attributed to two complementary components:

- **Geometry-Aware Modeling:** The GA module incorporates spherical coordinate information, frequency attention, and deformable convolution to adaptively model the geometric characteristics of ERP images.
  
- **Quality Context Representation:** The QCR module combines global contextual information with multi-scale local features to capture both holistic quality perception and localized distortion patterns.

With only **4.7M parameters** and approximately **1.5G FLOPs**, GAQC provides an efficient and effective framework for blind omnidirectional image quality assessment.

Detailed quantitative comparisons and experimental analyses can be found in our paper.

---

##  📝 Citation

If you find this repository useful for your research, please consider citing our work:

```bibtex
@article{yan2026gaqc,
  title={Efficient Blind Omnidirectional Image Quality Assessment: A Two-Dimensional Perspective},
  author={Yan, Jiebin and Li, Ziyi and Wu, Kangcheng and Zuo, Yifan and Fang, Chengyang and Fang, Yuming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026}
}
```

---
