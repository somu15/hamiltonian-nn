#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 22:15:19 2022

@author: dhulls
"""
import csv
import autograd.numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

input_dim1 = 20
file  = '/Users/dhulls/Desktop/German_Credit_20.csv'
data = np.zeros((1000,21))
count = 0
with open(file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if count > 0:
            for ii in np.arange(0,21,1):
                data[count-1,ii] = float(row[ii])
        count = count + 1

for ii in np.arange(1,21,1):
    data[:,ii] = (data[:,ii] - np.mean(data[:,ii])) / np.std(data[:,ii])

# data[:,2] = (data[:,2] - np.mean(data[:,2])) / np.std(data[:,2])
# data[:,5] = (data[:,5] - np.mean(data[:,5])) / np.std(data[:,5])
# data[:,13] = (data[:,13] - np.mean(data[:,13])) / np.std(data[:,13])

vals = np.zeros(100000)
for ii in np.arange(0,100000,1):
    
    param = norm().rvs(20)
    
    # term1 = 0.0
    # for ii in np.arange(0,1000,1):
    #     f_i = np.log(1+np.exp(np.sum(data[ii,1:21] * np.array(param).reshape(20))*data[ii,0])) + np.sum(np.array(param).reshape(20) * np.array(param).reshape(20))/(2000.0) #
    #     term1 = term1 + f_i
    
    term1 = np.sum(np.log(np.exp(np.sum(data[:,1:21] * np.array(param).reshape(20),axis=1)*data[:,0])+1)+np.sum(np.array(param).reshape(20) * np.array(param).reshape(20))/(2000.0))
    vals[ii] = term1