# Rock-Paper-Scissors Image Classification
*Created Using Python 3.7.11 and Tensorflow 2.5.0*

Image Classification of Rock-Paper-Scissors Hand Gesture Pictures using Convolutional Neural Network (CNN). 

## Full Code 
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N_tvqxojAXQhEyXElCzzHLIBx5E9eH4r?usp=sharing) 

## Library
This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [matplotlib](http://matplotlib.org/)
- [Keras](https://keras.io/)

You can just run it on Google Colab, but if you want to run it locally you will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included. 

## Dataset
### Dataset : [**rockpaperscissors.zip**](https://dicodingacademy.blob.core.windows.net/picodiploma/ml_pemula_academy/rockpaperscissors.zip)
The dataset used contains a total of 2188 hand gestures that make up,

1. 'Rock' (726 images)
2. 'Paper' (710 images)
3. 'Scissors' (752 images)

All images are RGB, 200x300 pixels in size, in .png format. The images are separated into three sub-folders named 'rock', 'paper' and 'scissors'. 

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
Some example of the dataset images (augmented):
![image](https://user-images.githubusercontent.com/63284781/127087418-f384e86e-e327-4e8a-9ee0-af7027f31766.png)
    
## Accuracy
Plotting the training and validation accuracy and loss of the trained model. I used callback to set the accuracy >= 97% (requirement for submission).

![image](https://user-images.githubusercontent.com/63284781/127088637-1998a815-bca6-45e6-9097-8923a41f368f.png)

## Testing
You can interactively selects an image file and test the prediction of the model. Example prediction: 

![image](https://user-images.githubusercontent.com/63284781/127088953-db50c518-5f23-45de-a4d6-ef81f217f1cc.png)
