# CheXNet Implementation

## CheXNet

- **Overview**  
 This is a Python3 (Pytorch) reimplementation of [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/). The model takes a chest X-ray image as input and outputs the probability of each thoracic disease along with a likelihood map of pathologies

## Prerequisites

- Python 3.4+
- [PyTorch](http://pytorch.org/) and its dependencies


## Model Training Overview


- Architecture: DenseNet-121 backbone with pre-trained ImageNet weights, adapted for multi-label classification with sigmoid activation
- Training: Binary Cross-Entropy loss with Adam optimizer
- Output: 14-class probability predictions for conditions like Pneumonia, Cardiomegaly, Atelectasis, etc.

 
## Reference

  Primary CheXNet paper: [https://arxiv.org/abs/2111.00595](https://arxiv.org/abs/1711.05225)

```
Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., ... & Ng, A. Y. (2017). Chexnet: Radiologist-level pneumonia detection on chest x-rays with deep learning. arXiv preprint arXiv:1711.05225.
```
