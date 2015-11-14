import cv2
import numpy as np 

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

def getStandardOutputVectors(n):
	return np.eye(n)

def transformImageForAnn(image):
	resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_NEAREST)
	normalized = resized / 255
	return normalized.flatten()

def createAnn(outputLayerDim, middleLayerDim = 128):
    ann = Sequential()
    ann.add(Dense(input_dim=784, output_dim=middleLayerDim,init="glorot_uniform"))
    ann.add(Activation("sigmoid"))
    ann.add(Dense(input_dim=middleLayerDim, output_dim=outputLayerDim,init="glorot_uniform"))
    ann.add(Activation("sigmoid"))
    return ann

def trainAnn(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, nb_epoch=500, batch_size=1, verbose = 0, shuffle=False, show_accuracy = False) 
    return ann

def getMaxIndex(arr):
	return max(enumerate(arr), key=lambda x: x[1])[0]

def classify(ann, inputs):
	outputs = ann.predict(np.array(inputs, np.float32))
	return [getMaxIndex(output) for output in outputs]

