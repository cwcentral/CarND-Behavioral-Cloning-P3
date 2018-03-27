import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
import cv2, numpy as np

#def resize_image(image_files):
	#for idx in range(0, len(image_files)):
		#im = cv2.resize(im, (224, 244))
	 #roi = img[60:140, :, :]
	# Resize the image
	#resize = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
	## Return the image sized as a 4D array
	#return np.resize(resize, (1, 224, 224, 3))



#use the vgg model from keras:
# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


lines = []
with open('./sim_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = './sim_data/IMG/' + filename
	image = cv2.resize(cv2.imread(current_path), (224, 224)).astype(np.float32)
    	#im[:,:,0] -= 103.939
    	#im[:,:,1] -= 116.779
    	#im[:,:,2] -= 123.68
    	#im = im.transpose((2,0,1))
	#image = np.expand_dims(image, axis=0)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

X_train = np.array(images)
Y_train = np.array(measurements)

model = VGG_16()

#X_train = resize_image(X_train_raw)

#plt.figure(figsize=(1,1))
#plt.imshow(X_train[0])
#import scipy.misc
#scipy.misc.imsave('outfile.png', X_train[0])

#model = Sequential()
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Flatten())
#model.add(Dense(1))

#model = Sequential()
#model.add(Convolution2D(32, 3, 3, input_shape=(160, 320, 3)))
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(32,3,3)))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout((0.75)))
#model.add(Activation('relu'))
#model.add(Flatten())
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dense(5))
#model.add(Activation('softmax'))

#X_normalized = np.array(X_train / 255.0 - 0.5 )
#from sklearn.preprocessing import LabelBinarizer
#label_binarizer = LabelBinarizer()
#y_one_hot = label_binarizer.fit_transform(Y_train)
#model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_one_hot, validation_split=0.2, shuffle=True, nb_epoch=10)

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

print("saving model")
model.save('model.h5')
print("saved")

