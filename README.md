# Rock-Paper-Scissors Image Classification
Image Classification of Rock-Paper-Scissors Hand Gesture Pictures using Convolutional Neural Network (CNN)

*Created Using Python 3.7.11 and Tensorflow 2.5.0*

### Full Code 
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N_tvqxojAXQhEyXElCzzHLIBx5E9eH4r?usp=sharing) 

### Import Library
```python
import zipfile 
import os

import numpy as np
import tensorflow as tf
from google.colab import files
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
```

## Dataset Preparation
### Dataset : [**rockpaperscissors.zip**](https://dicodingacademy.blob.core.windows.net/picodiploma/ml_pemula_academy/rockpaperscissors.zip)
The dataset used contains a total of 2188 hand gestures that make up,

1. 'Rock' (726 images)
2. 'Paper' (710 images)
3. 'Scissors' (752 images)

All images are RGB, 200x300 pixels in size, in .png format. The images are separated into three sub-folders named 'rock', 'paper' and 'scissors'. 

### Download Dataset
```python
# Download dataset to folder /tmp
!wget --no-check-certificate \
  https://dicodingacademy.blob.core.windows.net/picodiploma/ml_pemula_academy/rockpaperscissors.zip \
  -O /tmp/rockpaperscissors.zip
```

### Extracting Dataset

```python
# Extracting zip file
local_zip = '/tmp/rockpaperscissors.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

base_dir = '/tmp/rockpaperscissors/rps-cv-images'
```
The following is the structure of a directory file named “rockpaperscissors.zip”. We will use '/tmp/rockpaperscissors/rps-cv-images' as the base directory
```
/tmp/rockpaperscissors
 ├── 📂paper
 ├── 📂rock
 ├── 📂rps-cv-images
 │ ├── 📂paper
 │ │ ├── 🖼️image1.jpg...image(n).jpg
 │ ├── 📂rock
 │ ├── 📂scissors
 │ ├── 📃README_rps-cv-images.txt
 ├── 📂scissors
 ├── 📃README_rps-cv-images.txt
```

