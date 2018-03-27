**Behavioral Cloning Project**

This repository contains starting files for the Behavioral Cloning Project.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
(Files stored in jpg format-- NOT to be uploaded)

* Build, a convolution neural network in Keras that predicts steering angles from images
This is [found here](model.py) as required in the rubric points. This model includes 3 different approaches, using AlexNet,LeNet, and Nvidia. From testing, the Nvidia model proved best and fastest.

* Train and validate the model with a training and validation set
I used the data [sample test provided by the lesson](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).

I augmented the dataset by adding additional images and performing transfer learning using the simiulator. These images are not included in the git repo, but can be identified in the driving_log.csv file.

The udacity dataset was modified by removing the 1st line (header) of its driving_log.csv.

* Test that the model successfully drives around track one without leaving the road

The model that was created is [provided as the model.h5 file](model.h5).

* Summarize the results with a written report

A recording of the model in action, running in the simluator was created is [provided as the video file](video.mp4).

My project includes the following files:
* [model.py](model.py) containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network 
* This Readme.md summarizing the results

#### 1. Project Overview

This lesson is to use "real-world" data and apply deep learning techniques to predict the correct steering angle needed to drive a car around a track. Of course, we use a simulator to generate this data, augment this data, and test our deep learning model. The model created is to identify what is the track lane (classification) in order to calculate a proper steering angle.

### 2. Data Set Summary

The dataset shows a bias towards staying in the center of the lane as the car moves around the track. 
![](example.jpg?raw=true)


#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
