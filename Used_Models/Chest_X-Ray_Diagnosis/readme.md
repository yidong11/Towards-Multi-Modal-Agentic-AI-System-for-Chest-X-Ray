# Chest X-Ray Image Diagnosis with Stochastic Gradient Descent and Momentum Implementation

## Chest X-Ray Image Diagnosis with Stochastic Gradient Descent and Momentum

- **Overview**  
This repository implements a robust chest X-ray diagnostic system using deep learning with stochastic gradient descent and momentum optimization. It employed Transfer Learning to retrain a DenseNet model for the purpose of classifying X-ray images

## Prerequisites

- Python 3.6+
- Keras == 2.3.1
- tensorflow-gpu == 1.15.0


## Model Training Overview

### DenseNet121:

   We have used a pre-trained DenseNet121 model which we have loaded directly from Keras and then add two layers on top of it:
  

### Training:
- Architecture: Pre-trained DenseNet121 backbone with GlobalAveragePooling2D and sigmoid activation for 14-class multi-label prediction
- Optimization: Stochastic Gradient Descent (SGD) with momentum, providing superior convergence and stability compared to standard optimizers
- A pre-trained model namely `my_model_weight.h5` has already been made for testing purposes.
### Testing and Evaluating:
   
   The `my_model_weight.h5` file is used for loading the weights of the pre-trained model and is used for testing.  

   To use the pre-trained model and evaluate it, run all the cells of  `inference.ipynb` by pressing ctrl+enter at each cell
 
## Reference

  Primary Chest X-Ray Image Diagnosis with Stochastic Gradient Descent and Momentum paper: [https://link.springer.com/article/10.1007/s11042-024-19721-8](https://link.springer.com/article/10.1007/s11042-024-19721-8)

```
@article{banik2024robust,
  title={Robust stochastic gradient descent with momentum based framework for enhanced chest X-ray image diagnosis},
  author={Banik, Debajyoty},
  journal={Multimedia Tools and Applications},
  pages={1--24},
  year={2024},
  publisher={Springer}
}
```
