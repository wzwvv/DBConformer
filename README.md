<div align="center">
<h1>DBConformer</h1>
<h3>Dual-Branch Convolutional Transformer for EEG Decoding</h3>

[Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en)<sup>1</sup>, [Hongbin Wang](https://github.com/WangHongbinary)<sup>1</sup>, [Tianwang Jia](https://github.com/TianwangJia)<sup>1</sup>, [Xingyi He](https://github.com/BAY040210)<sup>1</sup>, [Siyang Li](https://scholar.google.com/citations?user=5GFZxIkAAAAJ&hl=en)<sup>1</sup>, and [Dongrui Wu](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)<sup>1 :email:</sup>

<sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

(<sup>:email:</sup>) Corresponding Author

[![DBConformer](https://img.shields.io/badge/Paper-DBConformer-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/2506.21140)&nbsp;

</div>

> This repository contains the implementation of our paper: **"DBConformer: Dual-Branch Convolutional Transformer for EEG Decoding"**, serving as a **benchmark codebase** for EEG decoding models. We implemented and fairly evaluated ten state-of-the-art EEG decoding models, including CNN-based, CNN-Transformer hybrid, and CNN-Mamba hybrid EEG decoding models.

## Overview
**DBConformer**, a **dual-branch convolutional Transformer** network tailored for EEG decoding:

- **T-Conformer**: Captures temporal dependencies
- **S-Conformer**: Models spatial patterns
- A **lightweight channel attention module** further refines spatial representations by assigning data-driven importance to EEG channels

<img width="1590" alt="image" src="https://github.com/user-attachments/assets/b4c0280f-f262-46c2-8f77-1ad649fde62a" />

## Features

- 🔀 **Dual-branch parallel design** for symmetric spatio-temporal modeling
- 🧩 **Plug-and-play channel attention** for data-driven channel weighting
- 📈 **Strong generalization** across CO, CV, and LOSO settings
- 💡 **Interpretable** aligned well with sensorimotor priors in MI
- 🧮 8× fewer parameters than large CNN-Transformer baselines (e.g., EEG Conformer)

Comparison of network architectures among CNNs (EEGNet, SCNN, DCNN, etc), traditional serial Conformers (EEG Conformer, CTNet, etc), and the proposed DBConformer. DBConformer has two branches that parallel capture temporal and spatial characteristics.

<div align="center">
  <img src="https://github.com/user-attachments/assets/4fe38406-61bc-45da-8a91-eceb1f4cf794" width="50%"/>
</div>

## Code Structure
```
DBConformer/
│
├── DBConformer_CO.py       # Main script for Chronological Order (CO) scenario
├── DBConformer_CV.py       # Main script for Cross-Validation (CV) scenario
├── DBConformer_LOSO.py     # Main script for Leave-One-Subject-Out (LOSO) scenario
│
├── models/                 # Model architectures (DBConformer and baselines)
│   ├── DBConformer.py      # Dual-branch Convolutional Transformer (Ours)
│   ├── EEGNet.py           # Classic CNN model
│   ├── SCNN.py             # Classic CNN model
│   ├── DCNN.py             # Classic CNN model
│   ├── FBCNet.py           # Frequency-aware CNN model
│   ├── ADFCNN.py           # Two-branch CNN model
│   ├── IFNet.py            # Frequency-aware CNN model
│   ├── EEGWaveNet.py       # Multi-scale CNN model
│   ├── SlimSeiz.py         # Serial CNN-Mamba baseline
│   ├── CTNet.py            # Serial CNN-Transformer baseline
│   └── EEGConformer.py     # Serial CNN-Transformer baseline
│
├── data/                   # Dataset
│   ├── BNCI2014001/
│   └── ...
│
├── utils/                  # Helper functions and common utilities
│   ├── data_utils.py           # EEG preprocessing, etc
│   ├── alg_utils.py           # Euclidean Alignment, etc
│   ├── network.py        # Backbone definition
│   └── ...
│
└── README.md
```

## Baselines
Ten EEG decoding models were reproduced and compared with the proposed DBConformer in this paper. DBConformer achieves the **state-of-the-art performance**.

- CNNs: EEGNet, SCNN, DCNN, FBCNet, ADFCNN, IFNet, EEGWaveNet
- Serial Conformers: CTNet, EEG Conformer 
- CNN-Mamba: SlimSeiz

<img width="1031" alt="image" src="https://github.com/user-attachments/assets/f0df1a55-b7e6-4865-8ca0-a4eab3067a33" />

## Datasets
DBConformer is evaluated on **MI classification** and **seizure detection** tasks. MI datasets can be downloaded from [MOABB](https://moabb.neurotechx.com), and [NICU dataset](https://zenodo.org/record/4940267). The processed BNCI2014001 dataset can be found in [MVCNet](https://github.com/wzwvv/MVCNet).

- Motor Imagery:
  - BNCI2014001
  - BNCI2014004
  - Zhou2016
  - Blankertz2007
  - BNCI2014002
- Seizure Detection:
  - CHSZ
  - NICU

## Experimental Scenarios
DBConformer supports three standard EEG decoding paradigms:

- **CO (Chronological Order):** Within-subject, EEG trials were partitioned strictly based on temporal sequence, with the first 80% used for training and the remaining 20% for testing.
- **CV (Cross-Validation):** Within-subject, stratified 5-fold validation. The data partitions were structured chronologically while maintaining class-balance.
- **LOSO (Leave-One-Subject-Out):** Cross-subject generalization evaluation. EEG trials from one subject were reserved for testing, while all other subjects’ trials were combined for training.

## Visualizations
### Effect of Dual-Branch Modeling
To further evaluate the impact of dual-branch architecture, we conducted feature visualization experiments using t-SNE. Features extracted by T-Conformer (temporal branch only) and DBConformer (dual-branch) were compared on four MI datasets.

<img width="1387" alt="image" src="https://github.com/user-attachments/assets/d6d8a8eb-bdf5-4b69-9dca-459f46d9cb8c" />

### Interpretability of Channel Attention
To investigate the interpretability of the proposed channel attention module, we visualized the attention scores assigned to each EEG channel across 32 trials (a batch) from four MI datasets. BNCI2014004 were excluded from this analysis, as it only contains C3, Cz, and C4 channels and therefore lacks spatial coverage for attention comparison.

<img width="1384" alt="image" src="https://github.com/user-attachments/assets/efebf73d-ea1c-46a8-8287-e5a2a0d352a7" />

---

## 📄 Citation
If you find this work helpful, please consider citing our paper:
```
@article{wang2025dbconformer,
      title={DBConformer: Dual-Branch Convolutional Transformer for EEG Decoding}, 
      author={Ziwei Wang, Hongbin Wang, Tianwang Jia, Xingyi He, Siyang Li, and Dongrui Wu},
      journal={arXiv preprint arXiv:2506.21140},
      year={2025}
}
```

## 🙌 Acknowledgments

Special thanks to the source code of EEG decoding models: [EEGNet](https://github.com/vlawhern/arl-eegmodels), [IFNet](https://github.com/Jiaheng-Wang/IFNet), [EEG Conformer](https://github.com/eeyhsong/EEG-Conformer), [FBCNet](https://github.com/ravikiran-mane/FBCNet), [CTNet](https://github.com/snailpt/CTNet), [ADFCNN](https://github.com/UM-Tao/ADFCNN-MI), [EEGWaveNet](https://github.com/IoBT-VISTEC/EEGWaveNet), and [SlimSeiz](https://github.com/guoruilu/SlimSeiz).

We appreciate your interest and patience. Feel free to raise issues or pull requests for questions or improvements.
