#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 22:08:02 2021

@author: dhulls
"""

from os import sys
import os
import pathlib
import numpy as np
import random
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.stats import uniform
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pickle
from statsmodels.distributions.empirical_distribution import ECDF
from autograd import grad

########## Multivariate Normal ###########

# ## Distribution parameters
# rho = 0.8

# ## Initialize parameters
# delta  = 0.3
# nsamples = 1000
# L = 20

# ## Potential energy function
# def UE(x,rho):
#     tmp = np.dot(np.tensordot(np.transpose(x), np.linalg.inv(np.array([[1, rho],[rho, 1]])), axes=1).reshape(2), x.reshape(2))
#     return tmp

# ## Gradient of potential energy function
# # def dUE(x,rho):
# #     tmp = np.tensordot(np.transpose(x), np.linalg.inv(np.array([[1, rho],[rho, 1]])), axes=1)
# #     return (tmp * 2)
# dUE = grad(UE)

# ## Kinetic energy function
# def KE(p):
#     tmp = np.dot(p.reshape(2),p.reshape(2)) / 2 
#     return tmp

# ## Initial conditions
# x = np.zeros((2,nsamples))
# x0 = 100*np.ones((2,1))
# x[:,0] = x0.reshape(2)

# normp = norm(loc=0,scale=1)
# unif = uniform()
# xsto = np.zeros((2,L+1))
# ## Leap frog integration
# t = 0
# while t < (nsamples-1):
#     t = t + 1 
#     p0 = normp.rvs((2,1))
    
#     pStar = p0 - delta/2 * dUE(x[:,t-1].reshape(2,1),rho).reshape(2,1)/2
#     xStar = x[:,t-1].reshape(2,1) + delta * pStar
#     xsto[:,0] = xStar.reshape(2)
    
#     for ii in np.arange(0,L):
#         pStar = pStar - delta * dUE(xStar.reshape(2,1),rho).reshape(2,1)/2
#         xStar = xStar + delta * pStar
#         xsto[:,ii+1] = xStar.reshape(2)
    
#     pStar = pStar - delta/2 * dUE(xStar.reshape(2,1),rho).reshape(2,1)/2
    
#     U0 = UE(x[:,t-1].reshape(2,1), rho)
#     UStar = UE(xStar.reshape(2,1), rho)
    
#     K0 = KE(p0.reshape(2,1))
#     KStar = KE(pStar.reshape(2,1))
    
#     alpha = np.minimum(1,np.exp((U0 + K0) - (UStar + KStar)))
#     if alpha > unif.rvs():
#         x[:,t] = xStar.reshape(2)
#     else:
#         x[:,t] = x[:,t-1]
        
# plt.scatter(x[0,1:1000],x[1,1:1000])
# # plt.plot(x[0,1:50],x[1,1:50], '-o')
# plt.show()

########## Arbitrary 2D ###########


## Initialize parameters
delta  = 0.01
nsamples = 1000
L = 2000

# rho = 0.8
## Potential energy function
def UE(x):
    # a = 1.15
    # b = 0.5
    # rho = 0.9
    # p = a * x[0,0]
    # q = x[1,0] / a + b * (x[0,0]**2 + a**2)
    # tmp1 = p**2/a**2
    # tmp2 = a**2 * (q - b * p**2/a**2 - b * a**2)**2
    # tmp3 = 2 * rho * (q - b * p**2/a**2 - b * a**2)
    # tmp4 = 1/(2 * (1-rho**2))
    # tmp = tmp4 * (tmp1 + tmp2 - tmp3)
    tmp = np.dot(np.tensordot(np.transpose(x), np.linalg.inv(np.array([[1, 1.98],[1.98, 4]])), axes=1).reshape(2), x.reshape(2))
    return tmp

## Gradient of potential energy function
dUE = grad(UE)

## Kinetic energy function
def KE(p):
    tmp = np.dot(p.reshape(2),p.reshape(2)) / 2 
    return tmp

## Initial conditions
x = np.zeros((2,nsamples))
x0 = 1*np.ones((2,1))
x[:,0] = x0.reshape(2)

normp = norm(loc=0,scale=1)
unif = uniform()
xsto = np.zeros((2,L+1))
## Leap frog integration
t = 0
while t < (nsamples-1):
    print(t)
    t = t + 1 
    p0 = normp.rvs((2,1))
    
    pStar = p0 - delta/2 * dUE(x[:,t-1].reshape(2,1)).reshape(2,1)/2
    xStar = x[:,t-1].reshape(2,1) + delta * pStar
    xsto[:,0] = xStar.reshape(2)
    
    for ii in np.arange(0,L):
        pStar = pStar - delta * dUE(xStar.reshape(2,1)).reshape(2,1)/2
        xStar = xStar + delta * pStar
        xsto[:,ii+1] = xStar.reshape(2)
    
    pStar = pStar - delta/2 * dUE(xStar.reshape(2,1)).reshape(2,1)/2
    
    U0 = UE(x[:,t-1].reshape(2,1))
    UStar = UE(xStar.reshape(2,1))
    
    K0 = KE(p0.reshape(2,1))
    KStar = KE(pStar.reshape(2,1))
    
    alpha = np.minimum(1,np.exp((U0 + K0) - (UStar + KStar)))
    if alpha > unif.rvs():
        x[:,t] = xStar.reshape(2)
    else:
        x[:,t] = x[:,t-1]

mean = np.zeros(2)
cov = np.asarray([[1, 1.98],
                  [1.98, 4]])
temp = np.random.multivariate_normal(mean, cov, size=1000)
plt.plot(temp[:, 0], temp[:, 1], '.')
plt.plot(x[0,1:1000],x[1,1:1000], 'r+')    
plt.show()