
# coding: utf-8

# In[3]:

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import tensorflow as tf
tf.python.control_flow_ops = tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import load_model
import h5py

def loadTrainOutputData(fileName):
	data = pd.read_csv(fileName)
	y = data['Prediction'].as_matrix()
	return y
def loadTrainDataIntoMatrix(fileName,noOfSamples):
	print 'Loading Data from ',fileName
	x = numpy.fromfile(fileName, dtype='uint8')
	x = x.reshape((noOfSamples,60,60))
	x[x<255]=0
	print 'Done'
	return x


trainDataLoc = '../Data/train_x.bin'
testDataLoc  = '../Data/test_x.bin'
trainYDataLoc = "../Data/train_y.csv"
trainDataNoOfSamples = 100000
testDataNoOfSamples = 20000

trainDataFeatures = loadTrainDataIntoMatrix(trainDataLoc,trainDataNoOfSamples)
testDataFeatures = loadTrainDataIntoMatrix(testDataLoc,testDataNoOfSamples)

trainDataOutput = loadTrainOutputData(trainYDataLoc)

X_train, X_validate, y_train, y_validate = train_test_split(trainDataFeatures, trainDataOutput, test_size=0.30, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, 60, 60).astype('float32')
X_validate = X_validate.reshape(X_validate.shape[0], 1, 60, 60).astype('float32')
#Normalize the data
X_train = X_train / 255
X_validate = X_validate / 255
#1-hot encoding for y
y_train = np_utils.to_categorical(y_train)
y_validate = np_utils.to_categorical(y_validate)
num_classes = y_validate.shape[1]


# In[8]:

def larger_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(64, 7, 7, border_mode='valid', input_shape=(1, 60, 60), activation='relu'))
#	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(64, 7, 7, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(64, 5, 5, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# In[13]:

# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_validate, y_validate), nb_epoch=15, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_validate, y_validate, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
model.save('../Data/model_ConvNeuralNets.h5')


# In[ ]:



