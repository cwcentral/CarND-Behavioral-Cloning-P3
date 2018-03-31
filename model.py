#
# P3 Behaviorial cloning
# Best run on a 1080 GPU
# 75% success chance on 760GTX mobile or less.
#
import os
import random
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,Adam
from keras.applications.vgg16 import VGG16
from keras.layers import Cropping2D


#
# VGG 16 model
#
def vgg_model():
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(224,224,3)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(4096, activation='relu')) #120
	model.add(Dense(4096, activation='relu')) #120
	model.add(Dense(1000, activation='softmax')) #120
	model.add(Dense(1))
	return model
	

#
# LeNet model
#
def lenet_model():
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Convolution2D(20, 5, 5, subsample=(2,2),activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(50, 5, 5, subsample=(2,2),activation='relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())
	model.add(Dense(120, activation='softmax')) #120
	model.add(Dense(84)) #84
	model.add(Dropout(0.5))
	model.add(Dense(1))
	#model.add(Dense(120, activation='relu')) #120
	#model.add(Dense(84, activation='relu')) #84
	#model.add(Dense(1, activation='softmax'))
	return model


#
# Nvidia model as described in Udacity lesson
# https://devblogs.nvidia.com/deep-learning-self-driving-cars/
# https://arxiv.org/pdf/1604.07316v1.pdf
#
def nvidia_model():
     
        # resize to nvidia model req's
	def resize_img(img):
		import tensorflow as tf
		return tf.image.resize_images(img, (66, 200))

	model = Sequential()

        # normalize
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

        # take a lot of sky out and focuses on the road
	model.add(Cropping2D(cropping=((57,5),(1,1))))

	model.add(Lambda(resize_img))

	model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation='relu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2),activation='relu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2),activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Dropout(0.5))  
	model.add(Flatten())
	model.add(Dropout(0.5)) 
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

#
# preprocess images
#
def process_image(img):
	# possible resize vgg, alexnet
	#img = cv2.resize(img,(200, 66), interpolation = cv2.INTER_AREA)

	# convert to YUV as nvidia paper describes
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	return img_yuv

#
# Load generator to prevent out of memory
# Based on the suggestion in the project videos
#
def generator(samples, batch_size=128):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                center_path = batch_sample[0]
                left_path = batch_sample[1]
                right_path = batch_sample[2]
                ctr_filename = center_path.split('/')[-1]
                current_path = './sim_data/IMG/' + ctr_filename
                left_current_path = './sim_data/IMG/' + left_path.split('/')[-1]
                right_current_path = './sim_data/IMG/' + right_path.split('/')[-1]

		# exploit the use of left and right images recorded
		# use the recommended 0.2 delta in camera view related to steering angle
                steering_center = float(batch_sample[3])
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction

		# add augmented data
                if abs(steering_center) <= 0.02:
                    images.extend([process_image(cv2.imread(current_path))])
                    measurements.extend([steering_center])
                else: 
                    images.extend([process_image(cv2.imread(current_path)), 
                		process_image(cv2.imread(left_current_path)),
                		process_image(cv2.imread(right_current_path))])
                    measurements.extend([steering_center, steering_left, steering_right])

	    # Augment with MORE data
            # trim image to only see section with road
            augmented_images = []
            augmented_measurements = []
            for (im, meas) in zip(images, measurements):
                augmented_images.append(im)
                augmented_measurements.append(meas)

                if True:
		    # Flip as suggested by lesson
                    augmented_images.append(cv2.flip(im,1))
                    augmented_measurements.append(-1.0*meas)
    
		    # blur
                    blur_in = cv2.GaussianBlur(im, (5,5), 10.0)
                    augmented_images.append(blur_in)
                    augmented_measurements.append(meas)

		    # rotate
                    #rot = random.randint(-10, 10)
                    #rows,cols,dep = im.shape
                    #M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
                    #im = cv2.warpAffine(im, M, (im.shape[0], im.shape[1]))
                    #augmented_images.append(np.resize(im, (160,320,3)))
                    #augmented_measurements.append(meas)
    
	    # assign and shuffle
            X_train = np.array(augmented_images)
            Y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, Y_train)
	
# Debug Helper
def show_histogram(dataset, tbins):
    plt.figure(figsize=(4,4))
    plt.hist(dataset, tbins)
    plt.show(block=True)

# Debug Helper
def show_image(img):
	cv2.imshow('ImageOutput', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#
# BEGIN main
#

# Get all measurements
lines = []
with open('./sim_data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # create a even distribution of staying on the road.
            # there's a lot of steering 0, so we take that out

            # if there's hard steering, we want this
            if abs(float(line[3])) >= 0.5:
                lines.append(line)
            # if there's minor steering, we want some of it
            elif abs(float(line[3])) >= 0.1:
                rot = random.randint(0, 10)
                if rot == 1:
                    lines.append(line)
            # if there's no steering, we want little of it
            else:
                rot = random.randint(0, 30)
                if rot == 1:
                    lines.append(line)

zeros = 0
idx = 0
vis_angles = []
new_lines = []
bins = np.arange(-1, 1, 0.1, dtype=float) # fixed bin size
for line in lines:
        vis_angles.append(float(line[3]))
show_histogram(vis_angles, bins)

# Split the sample csv into test and validation set at 80/20
print(len(lines))
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# create generators as recommended in rubric
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

# Choose your model/architecture
model = nvidia_model()

#model = lenet_model()
#model = vgg_model()

# compile with MSE and adam
model.compile(loss='mse', optimizer='adam')

# Run it
# Set batchsize = num of augmented images * set numnber that greater than sample set
SAM_SZ=5*192
# 15 epoches
EP=15
history_object = model.fit_generator(train_generator, samples_per_epoch=SAM_SZ, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=EP) 

# Save it
print("saving model")
model.save('model.h5')
print("saved")

# print history object keys
print(history_object.history.keys())

# plot the training and validation loss for each epoch and compare fitting
# as explained in lesson
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# DONE
