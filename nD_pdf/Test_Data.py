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
input_dim1 = 27

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

## Ref: 50; 2 and 200



# def get_args():
#     return {'input_dim': 54,
#          'hidden_dim': 100,
#          'learn_rate': 5e-4,
#          'nonlinearity': 'sine',
#          'total_steps': 25000,
#          'field_type': 'solenoidal',
#          'print_every': 200,
#          'name': 'ndpdf',
#          'use_rk4' : 'True',
#          'gridsize': 10,
#          'input_noise': 0.01,
#          'seed': 0,
#          'save_dir': './{}'.format(EXPERIMENT_DIR),
#          'fig_dir': './figures'}

# class ObjectView(object):
#     def __init__(self, d): self.__dict__ = d

# args = ObjectView(get_args())

###### Neural ODE params ########

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=200)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--input_dim', type=float, default=54)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)


# with torch.no_grad():
#     true_y = odeint(Lambda(), true_y0, t, method='dopri5')
true_y = odeint(Lambda(), true_y0, t, method='dopri5')

def get_batch():
    s = torch.from_numpy(np.array([363, 352, 631, 267, 372, 768, 593, 330, 974, 697, 922, 286, 906,
           238, 987, 778, 971, 972, 913, 727])) # np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False)
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class ODEFuncGrad(nn.Module):

    def __init__(self, init_values):
        super(ODEFuncGrad, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            nn.Linear(5, 2)
        )
        counter = 0
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.Parameter(init_values[counter])
                counter += 1
                m.bias = nn.Parameter(init_values[counter])
                counter += 1

    def forward(self, t, y):
        return self.net(y**3)

def convert_to_nn_tensor(init_first, nlayers):
    t1 = np.zeros((nlayers,2))
    counter = 0
    for ii in np.arange(0,nlayers,1):
        t1[ii,0] = init_first[counter]
        counter += 1
    for ii in np.arange(0,nlayers,1):
        t1[ii,1] = init_first[counter]
        counter += 1
    t1 = torch.tensor(t1)
    t2 = np.zeros(nlayers)
    for ii in np.arange(0,nlayers,1):
        t2[ii] = init_first[counter]
        counter += 1
    t2 = torch.tensor(t2)
    t3 = np.zeros((2,nlayers))
    for ii in np.arange(0,nlayers,1):
        t3[0,ii] = init_first[counter]
        counter += 1
    for ii in np.arange(0,nlayers,1):
        t3[1,ii] = init_first[counter]
        counter += 1
    t3 = torch.tensor(t3)
    t4 = np.zeros(2)
    t4[0] = init_first[counter]
    counter += 1
    t4[1] = init_first[counter]
    t4 = torch.tensor(t4)
    return (t1,t2,t3,t4)

def convert_to_array(tens, nlayers):
    init_first = np.zeros(2*nlayers+nlayers+nlayers*2+2)
    t1 = tens[0]
    counter = 0
    for ii in np.arange(0,nlayers,1):
        init_first[counter] = t1[ii,0]
        counter += 1
    for ii in np.arange(0,nlayers,1):
        init_first[counter] = t1[ii,1]
        counter += 1
    t2 = tens[1]
    for ii in np.arange(0,nlayers,1):
        init_first[counter] = t2[ii]
        counter += 1
    t3 = tens[2]
    for ii in np.arange(0,nlayers,1):
        init_first[counter] = t3[0,ii]
        counter += 1
    for ii in np.arange(0,nlayers,1):
        init_first[counter] = t3[1,ii]
        counter += 1
    t4 = tens[3]
    init_first[counter] = t4[0] 
    counter += 1
    init_first[counter] = t4[1]
    return init_first

batch_y0, batch_t, batch_y = get_batch()

def NODE_hamil(coords):

    #******** Neural ODE #********
    
    dic1 = np.split(coords,2*input_dim1)
    init_params = convert_to_nn_tensor(dic1[0:int(input_dim1)], 5)
    func = ODEFuncGrad(init_params).to(device)
    pred_y = odeint(func.float(), batch_y0, batch_t).to(device)
    loss = torch.mean(torch.abs(pred_y - batch_y))
    term1 = loss.detach().numpy()
    
    for ii in np.arange(0,2*input_dim1,1):
        term1 = term1 + 1*dic1[ii]**2/2
    
    return term1

def getgrad_NODE(coords):
    dic1 = np.split(coords,2*input_dim1)
    init_params = convert_to_nn_tensor(dic1[0:int(input_dim1)], 5)
    func = ODEFuncGrad(init_params).to(device)
    pred_y = odeint(func.float(), batch_y0, batch_t).to(device)
    loss = torch.mean(torch.abs(pred_y - batch_y))
    grad_tensor = torch.autograd.grad(outputs=loss, inputs=func.parameters())
    grad_req = np.zeros(2*input_dim1)
    grad_req[0:int(input_dim1)] = convert_to_array(grad_tensor, 5)
    grad_req[0:int(input_dim1)] = grad_req[0:int(input_dim1)] + np.array(dic1[0:int(input_dim1)]).reshape(input_dim1)
    grad_req[int(input_dim1):int(2*input_dim1)] = np.array(dic1[int(input_dim1):int(2*input_dim1)]).reshape(input_dim1)
    return grad_req

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
    q1, q2, p1, p2 = np.split(coords,4)
    sigma_inv = np.array([[1.,0.],[0.,1.]])
    term1 = 0.
    
    mu = np.array([3.,0.])
    y = np.array([q1-mu[0],q2-mu[1]])
    tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    
    mu = np.array([-3.,0.])
    y = np.array([q1-mu[0],q2-mu[1]])
    tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    
    mu = np.array([0.,3.])
    y = np.array([q1-mu[0],q2-mu[1]])
    tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    
    mu = np.array([0.,-3.])
    y = np.array([q1-mu[0],q2-mu[1]])
    tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    
    term1 = -np.log(term1)
    term2 = p1**2/2+p2**2/2
    H = term1 + term2

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

    # ******** 24D German Credit Data #********
    # input_dim1 = 24
    # file  = '/Users/dhulls/Desktop/German_Credit.csv'
    # data = np.zeros((1000,input_dim1+1))
    # count = 0
    # with open(file, 'r') as file:
    #     reader = csv.reader(file)
    #     for row in reader:
    #         if count > 0:
    #             for ii in np.arange(0,input_dim1+1,1):
    #                 data[count-1,ii] = float(row[ii])
    #         count = count + 1
    #
    # for ii in np.arange(0,input_dim1,1):
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
    # term1 = np.sum(np.log(np.exp(np.sum(data[:,0:24] * np.array(param).reshape(24),axis=1)*data[:,24])+1)+ np.sum(np.array(param).reshape(24) * np.array(param).reshape(24))/(2000.0))
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + 1*dic1[ii]**2/2
    # H = term1 + term2

    # ********* nD Heirarchical (https://crackedbassoon.com/writing/funneling) *********
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = dic1[0]**2/2
    # term1 = term1 - np.log(2 / (np.pi * (1 + np.exp(dic1[1])**2)))
    # for ii in np.arange(2,input_dim1,1):
    #     term1 = term1 + (dic1[ii] - dic1[0])**2 / (2 * np.exp(dic1[1])**2)
    # term2 = 0.0
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + dic1[ii]**2/2 # term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
    # H = term1 + term2

    # ******** nD Gaussian with Wishart covariance #********
    # dic1 = np.split(coords,2*input_dim1)
    # with open('/Users/dhulls/Desktop/outfile.txt', 'r') as f:
    #     sigma_inv = [[float(num) for num in line.split(',')] for line in f]
    # sigma_inv = np.matrix(sigma_inv)
    # term1 = np.array(0.5 * np.array(dic1[0:int(input_dim1)]).T * sigma_inv * np.array(dic1[0:int(input_dim1)])).reshape(1)
    # term2 = np.array(0.5 * np.array(dic1[int(input_dim1):int(2*input_dim1)]).T * np.matrix(np.eye(input_dim1)) * np.array(dic1[int(input_dim1):int(2*input_dim1)])).reshape(1)
    # H = term1 + term2

    return H

# coords = np.random.rand(40)
# dcoords = autograd.grad(func1)(coords)
# np.sum(np.log(np.exp(np.sum(data[:,1:21] * np.array(param).reshape(20),axis=1)*data[:,0])+1)+np.sum(np.array(param).reshape(20) * np.array(param).reshape(20))/(2000.0))
# data[data[:,0]==0,0]
