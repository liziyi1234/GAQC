GAQC: Efficient Blind Omnidirectional Image Quality Assessment

Introduction：
GAQC is an efficient omnidirectional image quality assessment method that directly takes ERP images as input without viewport prediction, formulating BOIQA as a BIQA problem.

Key Features:

Lightweight & Efficient: Only 4.7M parameters, 1.5G FLOPs

GA Module: Spherical coordinate encoding + frequency attention + deformable convolution for adaptive geometric correction

QCR Module: Global context fusion with multi-scale local features

Quick Start：

Installation：pip install torch torchvision einops timm scipy pandas pillow tqdm

Data Preparation

Training：python train.py

Inference

File Description：

GAQC.py：Main model (GA + QCR)

train.py：Training entry point

config.py：Configuration parameters

MyDataset.py：Data loader

utils.py： Utility functions

Citation：

@article{yan2026gaqc,

  title={Efficient Blind Omnidirectional Image Quality Assessment: A Two-Dimensional Perspective},
  
  author={Yan, Jiebin and Li, Ziyi and Wu, Kangcheng and Zuo, Yifan and Fang, Chengyang and Fang, Yuming},
  
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  
  year={2026}
  
}

