# MFSFNet: A Multi-Frequency Selection Fusion Network for Multi-modal Semantic Segmentation

This repository contains the official implementation of MFSFNet, a novel network for multimodal semantic segmentation.

1) We propose a low computational cost multi-frequency fusion framework for multi-modal semantic segmentation, which performs progressive integration of complementary information while maintaining modality-specific representations.

2) We propose an MFS module to select global complementary features in the frequency domain. It further extracts useful representations in the spatial domain to suppress modality-specific noise.

3) We propose an MFF module that simultaneously leverages multi-frequency representations and channel information to capture long-range dependencies and inter-channel relationships. Experiments conducted on the ISPRS Vaihingen and Potsdam datasets demonstrate that the proposed method achieves competitive semantic segmentation performance with lower computational complexity.
4) 
## Installation
1. Requirements
   
- Python 3.10.15	
- CUDA 12.1
- torch==1.13.0+cu117
- torchvision==0.14.0+cu117
- tqdm==4.66.4
- numpy==1.23.5
- pandas==2.0.1
- ipython==8.12.3

## Demo
To quickly test the MGFNet with randomly generated tensors, you can run the demo.py file. This allows you to verify the model functionality without requiring a dataset.
1. Ensure that the required dependencies are installed:
```
pip install -r requirements.txt
```
2. Run the demo script:
```
python demo.py
```

## Datasets
All datasets including ISPRS Potsdam, ISPRS Vaihingen can be downloaded [here](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets).
