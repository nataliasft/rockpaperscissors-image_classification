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


## Dataset Preprocessing
To variance in the training data, we used `ImageDataGenerator` function to augment the train Dataset.
### Augmenting Images
```python
train_datagen = ImageDataGenerator(
    rescale=(1/255.),              # normalize the image vector, by dividing with 255.
    width_shift_range=0.2,         # randomize shifting width in the range of 0.2
    height_shift_range=0.2,        # randomize shifting height in the range of 0.2
    zoom_range=0.2,                # randomize zoom in the range of 0.2
    shear_range=0.2,               # randomize shear in the range of 0.2
    rotation_range=20,             # randomize rotation in the range of 20 degree
    brightness_range=[0.8,1.2],    # randomize brightness in between 0.8 - 1.2
    horizontal_flip=True,          # randomly flipping the image
    validation_split = 0.4,        # divide training data 1314 samples, and validation data 874 samples
    fill_mode="nearest"            # use fill mode if dataset background are plain color
)
```

Using the previous image data generator object to label training data and validation data. 
```python
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size = (100, 150),     # rescale image menjadi ukuran 100x150 pixels
    class_mode = 'categorical',   # tipe label (>2)
    subset = 'training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size = (100, 150),
    class_mode = 'categorical',
    subset = 'validation'
)

print(train_generator.class_indices)
```
```
Found 1314 images belonging to 3 classes.
Found 874 images belonging to 3 classes.
{'paper': 0, 'rock': 1, 'scissors': 2}
```

### Plotting the Augmented image
```python
target_labels = next(os.walk(base_dir))[1]
target_labels.sort()
batch = next(train_generator)
batch_images = np.array(batch[0])
batch_labels = np.array(batch[1])

target_labels = np.asarray(target_labels)

plt.figure(figsize=(15,10))
for n, i in enumerate(np.arange(10)):
    ax = plt.subplot(3,5,n+1)
    plt.imshow(batch_images[i])
    plt.title(target_labels[np.where(batch_labels[i]==1)[0][0]])
    plt.axis('off')
```
![image](https://user-images.githubusercontent.com/63284781/127087418-f384e86e-e327-4e8a-9ee0-af7027f31766.png)
    
