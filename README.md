# Traffic Sign Recognition 
## Overview
In this project, I have Trained a convolutional neural networks (CNN) model so it can classify 43 different traffic signs. 
The model is trained on the German Traffic Sign Dataset. which contained more than 51k images with 43 different classes. The entire dataset is divided into train, test, and validation.  During preprocessing every image is reshaped into 32X32 pixels as well as they get converted into a grayscale image. <br>
To increase the accuracy Data Augmentation is used on the training set data. Augmenting the data is creating more images from the available images but with a slight alteration of the images.

## STATE OF ART USED

<img src="https://github.com/sid113/Traffic-sign-Recognition-/blob/master/demo/vgg16.png" width="700" height="400" />

Here I have used VGGnet16 state of art with train accuracy 99% and test accuracy 97%. Sometimes the model gets confuses while predicting speed limits e.g. for the Speed limit (30km/h) it shows Speed limit (60km/h).

## Getting Started

### Dependencies
This project requires Python 3.6 and the following Python libraries installed:

* NumPy
* SciPy
* scikit-learn
* TensorFlow
* PIL
* OpenCV
* Pygame 

### Setting up the project 
* Clone the repo
```
https://github.com/sid113/Traffic-sign-Recognition-.git
```
* Run following Command 
```
$ cd Traffic-sign-Recognition-/
$ python pygameui2.py
```
## Demonstration Video
<img src="https://github.com/sid113/Traffic-sign-Recognition-/blob/master/demo/demo%20video.gif" width="800" height="500" />

## Authors

Contributors names and contact info

* Siddhesh Pawar:&nbsp;siddeshpawar03@gmail.com <br>
