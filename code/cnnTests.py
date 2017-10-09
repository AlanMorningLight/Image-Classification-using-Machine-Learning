import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils impogedrt np_utils
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
from sklearn.metrics import classification_report

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

trainDataLoc = '../dataset/train_x.bin'
testDataLoc  = '../dataset/test_x.bin'
trainYDataLoc = "../dataset/train_y.csv"
trainDataNoOfSamples = 100000
testDataNoOfSamples = 20000

trainDataFeatures = loadTrainDataIntoMatrix(trainDataLoc,trainDataNoOfSamples)
testDataFeatures = loadTrainDataIntoMatrix(testDataLoc,testDataNoOfSamples)

trainDataOutput = loadTrainOutputData(trainYDataLoc)

X_train, X_validate, y_train, y_validate = train_test_split(trainDataFeatures, trainDataOutput, test_size=0.30, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, 60, 60).astype('float32')
X_validate = X_validate.reshape(X_validate.shape[0], 1, 60, 60).astype('float32')
#Normalize the data
print 'Normalizing Data...'
X_train = X_train / 255
X_validate = X_validate / 255
print 'Done'
#1-hot encoding for y
y_train = np_utils.to_categorical(y_train)
y_validate = np_utils.to_categorical(y_validate)

def arrangement1():
	model = Sequential()
	model.add(Convolution2D(64, 7, 7, border_mode='valid', input_shape=(1, 60, 60), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(64, 7, 7, border_mode='valid', input_shape=(1, 60, 60), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(64, 7, 7, border_mode='valid', input_shape=(1, 60, 60), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(19, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def arrangement2():
	model = Sequential()
	model.add(Convolution2D(64, 7, 7, border_mode='valid', input_shape=(1, 60, 60), activation='relu'))#54
	model.add(Convolution2D(64, 7, 7, activation='relu'))#48
	model.add(MaxPooling2D(pool_size=(2, 2)))#24
	model.add(Convolution2D(64, 5, 5, activation='relu'))#20
	model.add(Convolution2D(64, 3, 3, activation='relu'))#18
	model.add(MaxPooling2D(pool_size=(2, 2)))#9
	model.add(Convolution2D(64, 3, 3, activation='relu'))#7
	model.add(Convolution2D(64, 3, 3, activation='relu'))#5
	model.add(MaxPooling2D(pool_size=(2, 2)))#3
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(19, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# choose the model to run
model = arrangement1()
#model = arrangement2()
# Fit the model
print 'Starting to Train...'
model.fit(X_train, y_train, validation_data=(X_validate, y_validate), nb_epoch=15, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_validate, y_validate, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
#model.save('model_ConvNeuralNets.h5')

#Getting the precision,recall,f1-score metrics
print 'Calculating performance metrics:'
predictions = model.predict(X_validate)
answer =[]
for i in predictions:
    x = [z for z, a in enumerate(i) if a == max(i)]
    answer.append(x[0])
target_names = ['class 0', 'class 1', 'class 2','class 3','class 4', 'class 5', 'class 6','class 7','class 8', 'class 9', 'class 10','class 11','class 12', 'class 13', 'class 14','class 15','class 16', 'class 17', 'class 18']
print classification_report(y_validate, answer,target_names=target_names)

#Predicting on Test Data & creating the prediction file to submit on kaggle
print 'Predicting for the Test Set'
fileName = 'kagglePredictions.csv'
X_test = testDataFeatures.reshape(testDataFeatures.shape[0], 1, 60, 60).astype('float32')
X_test = X_test / 255
predictions = model.predict(X_test)
answer =[]
for i in predictions:
    x = [z for z, a in enumerate(i) if a == max(i)]
    answer.append(x[0])
count = 0
result = pd.DataFrame(columns=('Id', 'Prediction'))
for i in answer:
    result.loc[count]=["%d"%count,"%d"%i]
    count += 1
result.to_csv(fileName, index=False)
print 'Predictions written to file',fileName
#===========================End Of File======================================


