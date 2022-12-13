# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:20:07 2022

@author: f.ngang
"""

import numpy
import sklearn.discriminant_analysis
import time
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ds = numpy.loadtxt("HTRU_2.csv",dtype=numpy.float,delimiter=',',usecols=[0,1,2,3,4,5,6,7,8])
y = ds[:, -1]
X = ds[:, :-1]

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.25, random_state=25)

trainy = list(map(int, trainy))
testy = list(map(int, testy))

print("Training set size: ",numpy.size(trainX, 0))
print("Testing set size: ",numpy.size(testX, 0))
print("-------------------------------------------------")

#LDA

#trainX, testX, trainy, testy
start_Train_time = time.time()
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(trainX,trainy)
train_elapsed_time = time.time() - start_Train_time
start_Test_time = time.time()
predictionLDA = lda.predict(testX)
Test_elapsed_time = time.time() - start_Test_time
errorLDA = sum(predictionLDA != testy)
matrix = confusion_matrix(predictionLDA, testy, labels=[1.0, 0.0])

print("Linear Discriminant Analysis")
print(f"Confusion matrix:\n {matrix}")
print("LDA Train Time spent: ", train_elapsed_time)
print("LDA Test Time spent: ", Test_elapsed_time)
print("LDA Error: ", errorLDA)
print('LDA Accuracy score is {:.2f}%'.format(accuracy_score(testy, predictionLDA)*100))
print("---------------------------------------------------")

#SVM

#trainX, testX, trainy, testy
start_time = time.time()
clf = svm.SVC(kernel='linear', C=1)
clf.fit(trainX, trainy)
elapsed_time = time.time() - start_time
start_test_time = time.time()
predictionSVC = clf.predict(testX)
test_elapsed_time = time.time() - start_test_time
errorSVC = sum(predictionSVC != testy)
Matrix = confusion_matrix(predictionSVC, testy, labels=[1.0, 0.0])

print("Support Vector Machine")
print(f"Confusion matrix:\n {Matrix}")
print("SVC Train Time spent: ", elapsed_time)
print("SVC Test Time spent: ", test_elapsed_time)
print("SVC Error: ", errorSVC)
print('SVC Accuracy score is {:.2f}%'.format(accuracy_score(testy, predictionSVC)*100))
