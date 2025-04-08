# *Parkinsons Disease*
![image](https://github.com/user-attachments/assets/8577d6d3-3f14-4456-9ac2-70ee268daa00)

# 📑Project Title
### Design of An Efficient Forecast Model For Too Early Parkinson’s Disease Detection    
# 📌Description
Parkinson’s disease (PD) is a progressive neurological disorder that presents symptoms similar to various other conditions, making early and accurate diagnosis a challenging task. Common symptoms include tremors, muscle stiffness, slowed movements, and balance issues. Recognizing the critical need for early detection to ensure better patient outcomes, this study explores the dynamic patterns of handwritten documents as a diagnostic marker for PD.

Traditional machine learning approaches relying on manually crafted features have shown limited accuracy and performance. To address these challenges, this work introduces an advanced deep learning model optimized for the early diagnosis of Parkinson’s disease. The proposed model leverages a genetic algorithm combined with the K-Nearest Neighbor (KNN) technique for feature optimization, resulting in significantly improved predictive accuracy.

The methodology achieves detection accuracy greater than 95%, precision exceeding 98%, an area under the curve (AUC) above 0.90, and a remarkably low loss of 0.12. Additionally, an ensemble strategy integrating Tuned KNN and Random Forest classifiers achieves an unprecedented 100% accuracy, highlighting the strength of the combined approach.

This study presents a strong and innovative diagnostic tool that not only outperforms several state-of-the-art machine learning and deep learning models but also makes a substantial contribution to the field of early Parkinson’s disease detection.
# 📌Table of Contents
- [Project Overview](#project-0verview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Required Imports and Libraries](#Required-mports-and-Libraries)
- [Project Structure](#Project-Structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures Used](#Model-Architectures-Used)
- [Training Details](#training-details)
- [Performance Metrics](#Performance-Metrics)
## 📌Project Overview
This project proposes a deep learning model for the early detection of Parkinson’s disease using optimized features through a genetic algorithm and KNN.
It achieves over 95% accuracy, 98% precision, and very low loss, outperforming traditional methods.
An ensemble of Tuned KNN and Random Forest further boosts the model to 100% accuracy.
## 📂Dataset
Dataset Source: https://www.kaggle.com/datasets/kmader/parkinsons-drawings

## 🛠️Dependencies
Before running the project, ensure the following Python libraries are installed:

- `tensorflow`: For  image process

- `keras`:  For neural networks Model

- `matplotlib`: For data visualization.

- `numpy`: For handling numerical data.

- `pandas`: For data manipulation and analysis.

- `os`: For handling file operations.
- Torchvision: For image processing tasks.
- Opencv: For computer vision image.
- Flask: For lightweight web framework in Python.
- SQLite: For simple database functionality.


To install these dependencies, run the following command:

```sh
pip install tensorflow keras matplotlib numpy pandas os
```
## 📌Required Imports & Libraries
```sh
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
```
## 📁Project Structure
```sh
Parkinsons dieasae/
│── dataset/               # Contains the dataset images
│── models/                # Pre-trained models and trained weights
│── src/
│   │── import_libraries.py  # Importing required libraries
│   │── import_dataset.py    # Loading and handling dataset
│   │── preprocess.py        # Data preprocessing and augmentation
│   │── image_processing.py  # Image transformations
│   │── train.py             # Model training (ResNet50, DenseNet169, VGG19, Xception, DenseNet201)
│   │── user_signup.py       # User sign-up implementation
│   │── user_signin.py       # User sign-in implementation
│   │── user_input.py        # Handling user inputs
│   │── final_outcome.py     # Producing the final classification outcome
│── requirements.txt       # Dependencies for the project
│── README.md              # Project documentation
│── app.py                 # Flask/FastAPI for deployment (optional)
```
## 🔄Data Preprocessing
### Training Data Augmentation:
```sh
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=False
)
```
### Test Data Preprocessing:
```sh
test_datagen = ImageDataGenerator(rescale=1./255)
```
## 🏗️Model Architectures Used
We experimented with multiple deep learning models:

- `VGG16`: A widely used Convolutional Neural Network (CNN) architecture with 16 layers. It follows a simple and uniform design with small 3×3 convolution filters and is known for its high accuracy in image classification tasks.

- `ResNet50`: A 50-layer deep residual network that introduces skip connections (residual learning) to avoid the vanishing gradient problem. It allows training of very deep networks with improved performance.

- `DenseNet169`: A densely connected CNN with 169 layers, where each layer receives input from all previous layers. This promotes feature reuse and reduces the number of parameters compared to traditional deep networks.

- `DenseNet201`: A deeper variant of DenseNet with 201 layers, further improving accuracy and efficiency through better feature propagation and gradient flow.

- `Xception`: An extension of the Inception architecture that uses depthwise separable convolutions, significantly reducing computational cost while maintaining high performance. It achieved the best accuracy in our experiments.
## 🎯Training Details

- Batch Size: 32

- Loss Function: Categorical Crossentropy

- Optimizer: Stochastic Gradient Descent (SGD)

- Number of Epochs: 20

- Evaluation Metrics: Accuracy, F1-Score, Precision, Recall
  
## 📊Performance Metrics

- Best Model: Xception

- Metrics Used: Accuracy, Precision, Recall, F1-Score

- Plots: Yes, training and validation accuracy/loss curves are included.
  
  `Loss`
  <img src="Xception Plots.png" alt="Image description" width="800" height="400">>
  
