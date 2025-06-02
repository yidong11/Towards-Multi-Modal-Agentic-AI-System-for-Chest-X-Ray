# CheXNet Implementation

## CheXNet

- **Overview**  
 This is a Python3 (Pytorch) reimplementation of [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/). The model takes a chest X-ray image as input and outputs the probability of each thoracic disease along with a likelihood map of pathologies

## Prerequisites

- Python 3.4+
- [PyTorch](http://pytorch.org/) and its dependencies


- **Model Selection** 
  You can change the model in `./inference.py`

```
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model = xrv.models.DenseNet(weights="densenet121-res224-rsna") # RSNA Pneumonia Challenge
model = xrv.models.DenseNet(weights="densenet121-res224-nih") # NIH chest X-ray8
model = xrv.models.DenseNet(weights="densenet121-res224-pc") # PadChest (University of Alicante)
model = xrv.models.DenseNet(weights="densenet121-res224-chex") # CheXpert (Stanford)
model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb") # MIMIC-CXR (MIT)
model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch") # MIMIC-CXR (MIT)
model = xrv.baseline_models.jfhealthcare.DenseNet() # DenseNet121 from JF Healthcare for the CheXpert competition
model = xrv.baseline_models.chexpert.DenseNet(weights_zip="chexpert_weights.zip") # Official Stanford CheXpert model

```

- ## Reference

  Primary CheXNet paper: [https://arxiv.org/abs/2111.00595](https://arxiv.org/abs/1711.05225)

```
Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., ... & Ng, A. Y. (2017). Chexnet: Radiologist-level pneumonia detection on chest x-rays with deep learning. arXiv preprint arXiv:1711.05225.
```
