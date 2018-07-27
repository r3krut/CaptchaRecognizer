# Captcha Recognizer

Aim of this project is to recognize heavily distorted textual numeric captha images with the state of the art machinery of neural networks and machine learning.
We propose a method based on U-Net network architecture with different kind of feature extractors, such as VGG11 and VGG16.

Our results show sufficient quality of recorgnition, as well as the possibility to improve the result in the future.

## License

This project is based on [Robot Surgery Segmentation](https://github.com/ternaus/robot-surgery-segmentation/) project with heavy modifications.


## Dataset


The dataset consists of <N> images with 200x100 resolution, which were aquired from the Internet website.

IMAGE


Every image in dataset was pre-processed to remove noise and increase continuity of the symbols. Used operations: LIST OF OPERATIONS

IMAGE


After that each image was manually labeled with specific color, where color represents class of the image. 

IMAGE

Final optimization step consists from converting image to grayscale color space. 
_Note: Colors values were taken with precise care, such that after transforming from RGB color space to grayscale color space they mustn't collide._

IMAGE


## Method

Our work is based on U-Net architecture with VGG11 and VGG16 encoders.

As the final step, we used our own post-processing to work out image segmentation problem. This step involves: 

## Training

TRAINING PROCESS

## Results

RESULTS

