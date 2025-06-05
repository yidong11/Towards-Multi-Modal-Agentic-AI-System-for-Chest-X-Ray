# TXV Models Implementation

## TorchXRayVision

- **Overview**  
  TorchXRayVision is an open source software library for working with chest X-ray datasets and deep learning models. We apply it to our dataset and generate corresponding pathology probabilities. Source: [https://github.com/mlmed/torchxrayvision](https://github.com/mlmed/torchxrayvision)

- **Requirement** 

  Python 3.8 or later and install the package:

```
$ pip install torchxrayvision
```

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

  Primary TorchXRayVision paper: [https://arxiv.org/abs/2111.00595](https://arxiv.org/abs/2111.00595)

```
@inproceedings{Cohen2022xrv,
title = {{TorchXRayVision: A library of chest X-ray datasets and models}},
author = {Cohen, Joseph Paul and Viviano, Joseph D. and Bertin, Paul and Morrison, Paul and Torabian, Parsa and Guarrera, Matteo and Lungren, Matthew P and Chaudhari, Akshay and Brooks, Rupert and Hashir, Mohammad and Bertrand, Hadrien},
booktitle = {Medical Imaging with Deep Learning},
url = {https://github.com/mlmed/torchxrayvision},
arxivId = {2111.00595},
year = {2022}
}
```