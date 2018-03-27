# **Behavioral Cloning Project**

This repository contains software for the Behavioral Cloning Project.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior

(Files stored in jpg format-- NOT to be uploaded)
* Build, a convolution neural network in Keras that predicts steering angles from images

This is [found here](model.py) as required in the rubric points. This model includes 3 different approaches, using AlexNet,LeNet, and Nvidia. From testing, the Nvidia model proved best and fastest.
* Train and validate the model with a training and validation set

I used the data [sample set, provided by the lesson](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).
I augmented the dataset by adding additional images and performing transfer learning using the simiulator. These images are not included in the git repo, but can be identified in the driving_log.csv file.
Note: The udacity dataset was modified by removing the 1st line (header) of its driving_log.csv.
* Test that the model successfully drives around track one time without leaving the road & crashing.

The model that was created is [provided as the model.h5 file](model.h5).

A recording of the model in action, running in the simluator was created is [provided as the video file](video.mp4).

### Rubric Points (included files)
What is included in this project (git repo) are

* [model.py](model.py) containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network 
* This Readme.md summarizing the results

* The project is organized with the following directory structure:
    main
    |--model.py
    |--model.h5
    |--drive.py
    |--examples (not checked in)
    |--run1 (not checked in)
    |--sim_data
           |
           |-- driving_log.csv
           |-- IMG
           
        
#### 1. Project Overview

This lesson is to collect data and train a deep learning network that determines the correct steering angle needed to drive a car around a track. Of course, we use a simulator to generate this data, develop training/validation datasets, augment this data, and create/test our deep learning model. The model created identifies what is the 'track lane' (classification of features) in order to calculate a proper steering angle and stay centered on the track.

### 2. Data Set Summary

The dataset shows a bias towards staying in the center of the lane as the car moves around the track. This is good driving behavior. In manual mode, we can drive the car around and collect data (steering angle, images, throttle) from 3 points of view: center, left and right. That is the input to my CNN.

![](example.jpg?raw=true)

#### Exploratory visualization of the dataset and Data Set Preprocessing

Looking at the sample dataset from driving, we noticed that the majority of time the car is moving straight & centered: the steering angle is zero most of the time. This doesn't give my model the best chance to identify patterns of when the car is going completely off track. The dataset size originally contained *24109* samples. Hence, data augmentation is needed.

![](hist_figure.png?raw=true)

*Variation in Dataset*

Images captured of the driving sequence are in 320x160x3 BGR format. We used all images available, center, left and right. In order to augment the steering measurement, I used the suggested **value of +- 0.2 radians** added for the steering angle of each left (<0) and right image (>0). This is shown in model.py line 139.

![](sim_data/IMG/right_2016_12_01_13_46_38_947.jpg?raw=true)

*Data Sample*

To preprocess the images and make them more unique, I had to remove the sky and generic horizon features. Hence I cropped the image:

![](cropped.png?raw=true)

*Cropped Data Sample*

I then performed the following image processing techniques to enhance the dataset and increase it 3x:
* Image Flip
* image Blur
* Image Rotate +- 30 degrees  (as the car should never be upside down)

I also developed a training and validation datasets from the single sample set, and used a 80/20 split rule as suggested in the project lessons. 80% of the shuffled data was designated *test data* and 20% *validation data*. This is shown in model.py line 215.

#### Training Helpers

Lastly in order to smoothly process the data, which I found it required up to 8GB or memory--not available on my computing platform; I had to implement python generators, which allow me to batch read image data as it was being processed by the model. My batch size was 16 images (3x = 48 images). In each generator, I would:
* add to sample set center, left and right images
* added steering angle offset per left and right image
* perform image augemntation as described above.
* shuffle the dataset and send it to the CNN.

Due to the poor performance I found with the LeNet and AlexNet architectures, as well as initial underfitting from the Nvidia architecture, I added additional data by driving in the simulator in troubled areas to capture good driving technique. Applying simple *transferred learning* techniques, I added the images and steering samples to the dataset. I did concluded adding too much data (more than 2 seconds in each troubled section) was easy to occur and as a result cause the model to diverge. Though it cause good MSE, during testing, the vehicle would radnomly turn off the road. Not having a steering wheel or joystick to smoothly direct the car back to the center lane proved difficult.

### 3. Model Architecture and Training Strategy

Using Keras, I initially tested the LeNet Architecture, which ended up having the car immediately drive off the road during testing. During training on LeNet, though used successfully in the previouos project, I could not get the loss values under 6% as well as it was overfitting the data. I then tried AlexNet Architecture, and proved successful half way down the track, eventually hitting a wall or going off road. It  also consumed a lot of GPU processing though calculated and MSE under 3%. As in the lesson, the Nvidia Architecture was suggested and hence used that model with good success.

The Nvidia Architecture is described here:
https://devblogs.nvidia.com/deep-learning-self-driving-cars/

And a paper was published here:
https://arxiv.org/pdf/1604.07316v1.pdf

Nvidia's training pipeline is described here:
![](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/training-624x291.png?raw=true)

*Copyright Nvidia--2018*


The Nvidia uses a novel approach to its CNN training model as follows (In Keras format):

* Input 360x120 
* Image Crop ((47,0), (0,0)) -- output 360x73x3 (altered from Nvidia's 200x66x3) 
* Normalization
* Convolution+RELU activation -- Input 360x73x3, Output: 178x35x24
* Convolution+RELU -- Input 178x34x24, Output: 87x16x36
* Convolution+RELU -- Input 87x16x36, Output: 42x6x48
* Convolution+RELU -- Input 42x6x48, Output: 40x4x64
* Convolution+RELU -- Input 40x4x64, Output: 38x2x64
* Flatten (Fully Connect)
* Dense Fully Connected to 100
* Dense Fully Connected to 50
* Dense Fully Connected to 10
* Dense Fully Connected to 1

![](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png?raw=true)

*Copyright Nvidia--2018*


During model training, I used a loss function of MSE (mean square error) and the Adam Optimizer (learning rate not modified).

#### Training Results

##### Nvidia Model: Winner

Intially with no cropping, no data augmentation, the model training proved acceptable, with loss on both test and validation under 3%, but would converge quickly under 3 epochs, then diverge quickly. After the modification to the model & preprocessing, I was able to get loss under 2% within 7 epochs, but was seeing some underfitting. Also the car would sometimes rub against walls after a sharp turn. Hence I went back to the simulator and collected more data on those areas. I was able to train a model that provided under 2% loss error, good fitting and sufficient speed under 7 epochs.

* What are some problems with the architecture?

It appears underfitting is the main issue with this model. It can use more data. Lastly, though I did not use the Nvidia recommended 200x66 image size, I was able to speed up the process by a crop operation instead.

Another issue is GPU dependency. The Nvidia model must run on a high-end GPU to finish in sufficient time. I also got varying results on different GPUs when generating a model. On a GTX760, though I had loss errors under 2%, training would produce a bad model and the car would drive off randomly in turns, i.e. training multiple times produced inconsistent models. On a GTX1080 though, I got (fast training and) consistent models and results.

*MSE Loss from Training*

![](figure.png?raw=true)

### Testing results

Using the nvidia driving simulator, that used my trained model, to output steer angles based on the real-time view of the center camera at a constant speed, I was able to test my model by driving in real-time:

Nvidia Driving Simulator (drive.py)
![](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/inference-624x132.png?raw=true)

*Copyright Nvidia--2018*

The resulting video of a single lap using my model can be viewed here:
[video-result](video.mp4?raw=true) or [here (youtube)](https://youtu.be/272J__Cajds)

NOTE: The trained model was executed on the simulator in 320x240 mode, with fastest rendering quality setting on **track 1**.

### Improvements to Training this CNN

* Image warping and shifting can be done to the dataset. These methods proved beneficial in the previous project: Classifiers.
* Subtracting/replacing the road surface to a solid color would be beneficial to provide lane-texture consistency.
* Better methods to crop the image to a trapazoid shape would be beneficial as shown in Project 1: Find Lane Lines.
* Running on a top-end GPU prove more consistent model training "run over run", and of course, faster to train.
* Identifying edges of the road that have no lines, e.g. road barriers, dirt runoff, or bump strips) and drawing a line to enhance the boundary would be useful as a augmentation step before training.
* Recommend a steerring wheel input if possible (as mentioned in the project lesson's hints) as good driving behavior is actually, difficult.
* Inject newly collected data in the correct timeslot of the dataset--i.e. better transferred learning techniques.
* Possibly try the ResNet Architecture since it has proven to be more accurate.
