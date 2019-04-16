from NeuroEvolution import NeuroEvolution
import pandas as pd
import tensorflow as tf
import random
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer

data, target = load_breast_cancer(return_X_y = True)

#------------------------------SETTING VARIABLE-----------------------------

populationSize = 10
maxLayers = 7
maxNodes = 16
maxIterations = 5
threshold = 1

#---------------------------PRE PROCESSING DATA-----------------------------

#Handpick categorical and continuous features from the input dataset
categoricalAttr = pd.DataFrame()
continuousAttr = pd.DataFrame(data)

encodedInput = pd.DataFrame()

random.seed(5)
flag = 0


def embedAttribute(attribute):
    if not flag:
        weights = [[10 * random.random() for i in range(len(attribute[0]) - 1)] for j in range(len(attribute[0]))]
        flag = 1
    attribute = attribute @ np.array(weights)
    return attribute

def EntityEmbedding():
    for i in range(len(categoricalAttr.columns)):
        labelEncoder = preprocessing.LabelEncoder()
        labelTemp = labelEncoder.fit_transform(categoricalAttr.iloc[:,i])
        labelTemp = labelTemp.reshape(len(labelTemp), 1)
        #Embedding the One Hot Encoded feature
        sparseTemp = embedAttribute(tf.keras.utils.to_categorical(labelTemp))
        sparseTemp = pd.DataFrame(sparseTemp, columns = [(len(encodedInput.columns) + len(continuousAttr.columns) + i) for i in range(len(sparseTemp[0]))])
        print(sparseTemp)
        encodedInput = pd.concat([encodedInput, sparseTemp], axis = 1)

inputData = pd.concat([continuousAttr, encodedInput], axis = 1)

data  = inputData.values

#Normalizing the dataset
std_scale = preprocessing.StandardScaler().fit(data)
data = std_scale.transform(data)


#----------------------------MAIN-------------------------------------------

trainX = data[:-50]
trainY = target[:-50]

testX = data[-50:]
testY = target[-50:]

Evolutor = NeuroEvolution(populationSize, maxLayers, maxNodes, maxIterations, threshold)
Evolutor.run(trainX, trainY, testX, testY, 6)
