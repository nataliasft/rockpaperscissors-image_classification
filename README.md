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
    
## Building the Model
Building CNN architecture
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (100, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(3, activation = 'softmax')
])
```
Using Categorical Loss Entropy loss function and Adam optimizer. 
```python
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = tf.optimizers.Adam(),
    metrics = ['accuracy']               
)
```

## Training Model
### Make a Callback
Stop training after >=97% accuracy (Requirement for Dicoding Project Submission)
```python
accuracy_threshold = 97e-2              
class accuracy_callbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        if logs.get('accuracy') >= accuracy_threshold:       # berhenti jika accuracy mencapai >= 0.97
            print('\nFor Epoch', epoch, '\nAccuracy has reach = %2.2f%%' %(logs['accuracy']*100), 'training has been stopped.')
            self.model.stop_training = True
```
### Training 
```python
history = model.fit(
    train_generator, 
    steps_per_epoch = 20, 
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = 5,
    verbose = 2,
    callbacks = [accuracy_callbacks()]
)
```
 <details> 
   <summary>
Training Result
   </summary>
   <pre>
Epoch 1/30
20/20 - 22s - loss: 1.0967 - accuracy: 0.3738 - val_loss: 1.0981 - val_accuracy: 0.3250
Epoch 2/30
20/20 - 22s - loss: 1.0405 - accuracy: 0.4609 - val_loss: 1.1140 - val_accuracy: 0.4375
Epoch 3/30
20/20 - 22s - loss: 0.8962 - accuracy: 0.5922 - val_loss: 0.9440 - val_accuracy: 0.5188
Epoch 4/30
20/20 - 22s - loss: 0.6882 - accuracy: 0.7312 - val_loss: 0.7002 - val_accuracy: 0.7125
Epoch 5/30
20/20 - 21s - loss: 0.5029 - accuracy: 0.8033 - val_loss: 0.5554 - val_accuracy: 0.7937
Epoch 6/30
20/20 - 22s - loss: 0.4564 - accuracy: 0.8078 - val_loss: 0.3559 - val_accuracy: 0.9000
Epoch 7/30
20/20 - 22s - loss: 0.4075 - accuracy: 0.8641 - val_loss: 0.3163 - val_accuracy: 0.9250
Epoch 8/30
20/20 - 22s - loss: 0.2759 - accuracy: 0.8969 - val_loss: 0.2096 - val_accuracy: 0.9125
Epoch 9/30
20/20 - 21s - loss: 0.2526 - accuracy: 0.9164 - val_loss: 0.2253 - val_accuracy: 0.8938
Epoch 10/30
20/20 - 22s - loss: 0.1980 - accuracy: 0.9297 - val_loss: 0.2325 - val_accuracy: 0.9250
Epoch 11/30
20/20 - 22s - loss: 0.1762 - accuracy: 0.9453 - val_loss: 0.1324 - val_accuracy: 0.9750
Epoch 12/30
20/20 - 22s - loss: 0.1238 - accuracy: 0.9625 - val_loss: 0.1881 - val_accuracy: 0.9250
Epoch 13/30
20/20 - 21s - loss: 0.1562 - accuracy: 0.9541 - val_loss: 0.1796 - val_accuracy: 0.9563
Epoch 14/30
20/20 - 21s - loss: 0.1530 - accuracy: 0.9443 - val_loss: 0.1182 - val_accuracy: 0.9563
Epoch 15/30
20/20 - 22s - loss: 0.0907 - accuracy: 0.9625 - val_loss: 0.1922 - val_accuracy: 0.9500
Epoch 16/30
20/20 - 21s - loss: 0.1095 - accuracy: 0.9656 - val_loss: 0.1615 - val_accuracy: 0.9625
Epoch 17/30
20/20 - 21s - loss: 0.1437 - accuracy: 0.9639 - val_loss: 0.2331 - val_accuracy: 0.9062
Epoch 18/30
20/20 - 21s - loss: 0.1307 - accuracy: 0.9557 - val_loss: 0.0891 - val_accuracy: 0.9500
Epoch 19/30
20/20 - 22s - loss: 0.0965 - accuracy: 0.9703 - val_loss: 0.0950 - val_accuracy: 0.9688

For Epoch 18 
Accuracy has reach = 97.03% training has been stopped.
   </pre>
</details>

### Accuracy
Plotting the training and validation accuracy and loss
```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

# Training and validation accuracy
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training dan validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()

# Training and validation loss
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training dan validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()

plt.show()
```
![image](https://user-images.githubusercontent.com/63284781/127088637-1998a815-bca6-45e6-9097-8923a41f368f.png)

## Testing
Interactively selects an image file then resizes the image and converts it into a numpy array. Example prediction of the model: 
```python
uploaded = files.upload()

for fn in uploaded.keys():
  # predict images
  path = fn
  img_source = image.load_img(path, target_size = (100, 150))
  imgplot = plt.imshow(img_source)
  x = image.img_to_array(img_source)
  x = np.expand_dims(x, axis = 0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size = 10)

  print(fn)
  if classes[0, 0] == 1:
    print('paper')
  elif classes[0, 1] == 1:
    print('rock')
  elif classes[0, 2] == 1:
    print('scissors')
```

![image](https://user-images.githubusercontent.com/63284781/127088953-db50c518-5f23-45de-a4d6-ef81f217f1cc.png)
