<div align="center">
<h1>DBConformer </h1>
<h3>Dual-Branch Convolutional Transformer for EEG Decoding</h3>

[Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en), Hongbin Wang, Tianwang Jia, Xingyi He, [Siyang Li](https://scholar.google.com/citations?user=5GFZxIkAAAAJ&hl=en), and [Dongrui Wu*](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)

</div>

> This repository contains the implementation of our paper: **"DBConformer: Dual-Branch Convolutional Transformer for EEG Decoding"**, serving as a **strong benchmark codebase** for EEG decoding tasks. We implemented and fairly evaluated ten state-of-the-art EEG decoding models, including CNN-based, CNN-Transformer hybrid, and CNN-Mamba hybrid. 

---

## 🧠 Overview
**DBConformer**, a **dual-branch convolutional Transformer** network tailored for EEG decoding:

- **T-Conformer**: Captures long-range temporal dependencies
- **S-Conformer**: Models inter-channel interactions
- A **lightweight channel attention module** further refines spatial representations by assigning data-driven importance to EEG channels

<img width="1590" alt="image" src="https://github.com/user-attachments/assets/b4c0280f-f262-46c2-8f77-1ad649fde62a" />

## 📦 Features

- 🔀 **Dual-branch parallel design** for symmetric spatio-temporal modeling
- 🧩 **Plug-and-play channel attention** for data-driven channel weighting
- 📈 **Strong generalization** across CO, CV, and LOSO settings
- 💡 **Interpretable** aligned well with sensorimotor priors in MI
- 🧮 8× fewer parameters than large CNN-Transformer baselines (e.g., EEG Conformer)

---

## 📂Code Structure
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

## 📊 Baselines
Ten EEG decoding models were reproduced and compared with the proposed DBConformer in this paper. DBConformer achieves the **state-of-the-art performance**.

- CNNs: EEGNet, SCNN, DCNN, FBCNet, ADFCNN, IFNet, EEGWaveNet
- Serial Conformers: CTNet, EEG Conformer 
- CNN-Mamba: SlimSeiz

## 📂 Datasets
DBConformer is evaluated on **MI classification** and **seizure detection** tasks.
- Motor Imagery:
  - BNCI2014001
  - BNCI2014004
  - Zhou2016
  - Blankertz2007
  - BNCI2014002
- Seizure Detection:
  - CHSZ
  - NICU

## 🧪 Experimental Scenarios
DBConformer supports three standard EEG decoding paradigms:

- **CO (Chronological Order):** Within-subject, time-based data split
- **CV (Cross-Validation):** Within-subject, stratified 5-fold validation
- **LOSO (Leave-One-Subject-Out):** Cross-subject generalization evaluation

---

## 📄 Citation
If you find this work helpful, please consider citing our paper:
```
@article{wang2025dbconformer,
      title={DBConformer: Dual-Branch Convolutional Transformer for EEG Decoding}, 
      author={Ziwei Wang, Hongbin Wang, Tianwang Jia, Xingyi He, Siyang Li, and Dongrui Wu},
      year={2025}
}
```
