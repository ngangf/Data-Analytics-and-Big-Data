# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:21:13 2022

@author: f.ngang
"""

# Question 1

import pandas
import numpy

dataA = pandas.read_excel('dataA.xlsx', header=None)

datanpyA = pandas.DataFrame.to_numpy(dataA)

meanA = numpy.mean(datanpyA,axis = 0)

qA = datanpyA - meanA

shapeA = datanpyA.shape
covA = numpy.dot(numpy.transpose(datanpyA - meanA),(datanpyA - meanA))/shapeA[0]

print('The mean of dataA is:')
print(meanA)
print('\n')

print('The covariance matrix of dataA is:')
print(covA)
print('\n')

#Question 2

dataB = pandas.read_excel('dataB.xlsx', header=None)

datanpyB = pandas.DataFrame.to_numpy(dataB)

meanB = numpy.mean(datanpyB,axis = 0)

qB = datanpyB - meanB

shapeB = datanpyB.shape
covB = numpy.dot(numpy.transpose(datanpyB - meanB),(datanpyB - meanB))/shapeB[0]

print('The mean of dataB is:')
print(meanB)
print('\n')

print('The covariance matrix of dataB is:')
print(covB)