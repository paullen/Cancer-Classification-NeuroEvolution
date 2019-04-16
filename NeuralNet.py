import os
import tensorflow as tf
import numpy as np

#Prevent Tensorflow messages from showing up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Make a neural network with properties specified
def makeDNN(numOfLayers, numOfNeurons, activationFunc):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())
    for i in range(numOfLayers):
        model.add(tf.keras.layers.Dense(numOfNeurons, activation = activationFunc))
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

#Single individual class definition
class NeuralNet:

    def __init__(self, layers, nodes, activation):
        self.layers = layers
        self.nodes = nodes
        self._testAccuracy = 0
        self._testLoss = 1
        self.activation = activation
        self.model = makeDNN(layers, nodes, activation)

    def setFitness(self, testX, testY):
        self._testLoss, self._testAccuracy = self.model.evaluate(testX, testY)

    def getAccuracy(self):
        return self._testAccuracy

    def getLoss(self):
        return self._testLoss

    def train(self, trainX, trainY, epoch):
        self.model.fit(trainX, trainY, epochs = epoch)
