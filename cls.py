import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder # Метод кодирования тестовых лейблов
from sklearn.model_selection import train_test_split

class CLS():
    def __init__(self, fileName):
        def readText(fileName):
            f = open(fileName, 'r', encoding='utf-8')
            text = f.read()
            text = text.replace("\n", " ")
            return text

        dirs = os.listdir('./Books')

        self.className = ["О. Генри", "Стругацкие", "Булгаков", "Саймак", "Фрай", "Брэдберри"]
        self.nClasses = len(self.className)

        self.trainText = []
        self.testText = []
        self.output_text = []
        for i in self.className:
            for j in dirs:
                if i in j:
                    if 'Обучающая' in j:
                        self.trainText.append(readText('./Books/' + j))
                        self.output_text.append(j+'добавлен в обучающую выборку')
                    if 'Тестовая' in j:
                        self.testText.append(readText('./Books/' + j))
                        self.output_text.append(j+'добавлен в тестовую выборку')

    def getSetFromIndexes(self, wordIndexes, xLen, step):
        xSample = []
        wordsLen = len(wordIndexes)
        index = 0
        while (index + xLen <= wordsLen):
            xSample.append(wordIndexes[index:index+xLen]) 
            index += step 
        return xSample
    
    def createSetsMultiClasses(self, wordIndexes, xLen, step):
        nClasses = len(wordIndexes) 
        classesXSamples = [] 
        for wI in wordIndexes: 
            classesXSamples.append(self.getSetFromIndexes(wI, xLen, step)) 
            xSamples = [] 
            ySamples = [] 
        for t in range(nClasses): 
            xT = classesXSamples[t] 
            for i in range(len(xT)): 
                xSamples.append(xT[i]) 
                ySamples.append(utils.to_categorical(t, nClasses)) 

        xSamples = np.array(xSamples) 
        ySamples = np.array(ySamples) 
        return (xSamples, ySamples)
    
    def createTrainData(self, maxWordsCount = 15000, xLen = 1000, step = 100):
        self.tokenizer = Tokenizer(num_words=maxWordsCount, filters = '')
        self.tokenizer.fit_on_texts(self.trainText)

        trainWordIndexes = self.tokenizer.texts_to_sequences(self.trainText)
        testWordIndexes = self.tokenizer.texts_to_sequences(self.testText)

        self.xTrain, self.yTrain = self.createSetsMultiClasses(trainWordIndexes, xLen, step)
        self.xTest, self.yTest = self.createSetsMultiClasses(testWordIndexes, xLen, step)

        self.xTrain01 = self.tokenizer.sequences_to_matrix(self.xTrain.tolist())
        self.xTest01 = self.tokenizer.sequences_to_matrix(self.xTest.tolist())

        return self.xTrain, self.xTrain01, self.yTrain, self.xTest, self.xTest01, self.yTest
    
    def createModel(self, maxWordsCount, nneurons, nlayers, factiv, dropout_rate):

        self.maxWordsCount = maxWordsCount
        self.nneurons = nneurons
        self.nlayers = nlayers
        self.factiv = factiv
        self.dropout_rate = dropout_rate

        self.model = Sequential()
        for i in range(nlayers):
            self.model.add(Dense(nneurons, input_dim = maxWordsCount, activation=factiv))
            self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(6, activation='sigmoid'))
        self.model.summary()
        self.model.compile(optimizer=Adam(learning_rate=0.0001), # 'adam'
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        return self.model
    
    def fitModel(self, epochs=5, batch_size=32, verbose=1):
        history = self.model.fit(self.xTrain01, 
                        self.yTrain, 
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(self.xTest01, self.yTest),
                        verbose=verbose)
        return history
    
    def testPrediction(self, testing_text):
        testWordIndexes = self.tokenizer.texts_to_sequences(testing_text)
        test01 = self.tokenizer.sequences_to_matrix(testWordIndexes)
        currPred = self.model.predict(test01)
        currOut = np.argmax(currPred, axis=1)
        evVal = []
        for j in range(self.nClasses):
            evVal.append(len(currOut[currOut==j])/len(test01))
        recognizedClass = np.argmax(evVal)
        return self.className[recognizedClass]