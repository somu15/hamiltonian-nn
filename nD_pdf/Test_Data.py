#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 16:19:28 2022

@author: dhulls
"""
import csv
import autograd.numpy as np
import autograd
import matplotlib.pyplot as plt
input_dim1 = 3

def func1(coords):
    # print(coords)
    #******** 1D Gaussian Mixture #********
    # q, p = np.split(coords,2)
    # mu1 = 1.0
    # mu2 = -1.0
    # sigma = 0.35
    # term1 = -np.log(0.5*(np.exp(-(q-mu1)**2/(2*sigma**2)))+0.5*(np.exp(-(q-mu2)**2/(2*sigma**2))))
    # H = term1 + p**2/2 # Normal PDF

    # #******** 2D Gaussian Four Mixtures #********
    # q1, q2, p1, p2 = np.split(coords,4)
    # sigma_inv = np.array([[1.,0.],[0.,1.]])
    # term1 = 0.
    #
    # mu = np.array([3.,0.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    #
    # mu = np.array([-3.,0.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    #
    # mu = np.array([0.,3.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    #
    # mu = np.array([0.,-3.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    #
    # term1 = -np.log(term1)
    # term2 = p1**2/2+p2**2/2
    # H = term1 + term2

    #******** 2D Highly Correlated Gaussian #********
    # q1, q2, p1, p2 = np.split(coords,4)
    # sigma_inv = np.array([[50.25125628,-24.87437186],[-24.87437186,12.56281407]])
    # term1 = 0.
    #
    # mu = np.array([0.,0.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    #
    # term1 = -np.log(term1)
    # term2 = p1**2/2+p2**2/2
    # H = term1 + term2

    # ******** 20D German Credit Data #********
    # input_dim1 = 20
    # file  = '/Users/dhulls/Desktop/German_Credit_20.csv'
    # data = np.zeros((1000,21))
    # count = 0
    # with open(file, 'r') as file:
    #     reader = csv.reader(file)
    #     for row in reader:
    #         if count > 0:
    #             for ii in np.arange(0,21,1):
    #                 data[count-1,ii] = float(row[ii])
    #         count = count + 1
    # # data[:,2] = (data[:,2] - np.mean(data[:,2])) / np.std(data[:,2])
    # # data[:,5] = (data[:,5] - np.mean(data[:,5])) / np.std(data[:,5])
    # # data[:,13] = (data[:,13] - np.mean(data[:,13])) / np.std(data[:,13])
    # for ii in np.arange(1,21,1):
    #     data[:,ii] = (data[:,ii] - np.mean(data[:,ii])) / np.std(data[:,ii])
    #
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = 0.0
    # param = dic1[0:input_dim1]
    # # print(param)
    # # for ii in np.arange(0,1000,1):
    # #     f_i = np.log(1+np.exp(np.sum(data[ii,1:21] * np.array(param).reshape(20))*data[ii,0])) + np.sum(np.array(param).reshape(20) * np.array(param).reshape(20))/(2000.0) #
    # #     term1 = term1 + f_i
    # term2 = 0.0
    # term1 = np.sum(np.log(np.exp(np.sum(data[:,1:21] * np.array(param).reshape(20),axis=1)*data[:,0])+1)+ np.sum(np.array(param).reshape(20) * np.array(param).reshape(20))/(2000.0))
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + 1*dic1[ii]**2/2
    # H = term1 + term2

    # ********* nD Heirarchical (https://crackedbassoon.com/writing/funneling) *********
    dic1 = np.split(coords,2*input_dim1)
    term1 = dic1[0]**2/2
    term1 = term1 - np.log(2 / (np.pi * (1 + np.exp(dic1[1])**2)))
    for ii in np.arange(2,input_dim1,1):
        term1 = term1 + (dic1[ii] - dic1[0])**2 / (2 * np.exp(dic1[1])**2)
    term2 = 0.0
    for ii in np.arange(input_dim1,2*input_dim1,1):
        term2 = term2 + dic1[ii]**2/2 # term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
    H = term1 + term2

    return H

# coords = np.random.rand(40)
# dcoords = autograd.grad(func1)(coords)
# np.sum(np.log(np.exp(np.sum(data[:,1:21] * np.array(param).reshape(20),axis=1)*data[:,0])+1)+np.sum(np.array(param).reshape(20) * np.array(param).reshape(20))/(2000.0))
# data[data[:,0]==0,0]
