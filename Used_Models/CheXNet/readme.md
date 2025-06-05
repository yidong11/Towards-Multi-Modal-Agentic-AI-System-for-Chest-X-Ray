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
@article{rajpurkar2017chexnet,
  title={Chexnet: Radiologist-level pneumonia detection on chest x-rays with deep learning},
  author={Rajpurkar, Pranav and Irvin, Jeremy and Zhu, Kaylie and Yang, Brandon and Mehta, Hershel and Duan, Tony and Ding, Daisy and Bagul, Aarti and Langlotz, Curtis and Shpanskaya, Katie and others},
  journal={arXiv preprint arXiv:1711.05225},
  year={2017}
}
```
