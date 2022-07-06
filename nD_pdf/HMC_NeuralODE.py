#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:35:19 2022

@author: dhulls
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 2021

@author: dhulls
"""

input_dim1 = 27
import torch, time, sys
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from statsmodels.distributions.empirical_distribution import ECDF
import tensorflow as tf
import tensorflow_probability as tfp

EXPERIMENT_DIR = './nD_pdf'
sys.path.append(EXPERIMENT_DIR)

import random
from scipy.stats import norm
from scipy.stats import uniform
import pandas as pd
from pandas.plotting import scatter_matrix

DPI = 300
FORMAT = 'pdf'
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2
RK4 = ''

import os
import argparse

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
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

######## HMC code ########

def leapfrog ( dydt, tspan, y0, n, dim ):
  t0 = tspan[0]
  tstop = tspan[1]
  dt = ( tstop - t0 ) / n

  t = np.zeros ( n + 1 )
  y = np.zeros ( [dim, n + 1] )

  for i in range ( 0, n + 1 ):

    if ( i == 0 ):
      t[0]   = t0
      for j in range ( 0, dim ):
          y[j,0] = y0[j]
      anew   = dydt ( t, y[:,i] ) 
    else:
      t[i]   = t[i-1] + dt
      aold   = anew
      for j in range ( 0, int(dim/2) ):
          y[j,i] = y[j,i-1] + dt * ( y[(j+int(dim/2)),i-1] + 0.5 * dt * aold[(j+int(dim/2))] )
      anew   = dydt ( t, y[:,i] ) 
      for j in range ( 0, int(dim/2) ):
          y[(j+int(dim/2)),i] = y[(j+int(dim/2)),i-1] + 0.5 * dt * ( aold[(j+int(dim/2))] + anew[(j+int(dim/2))] )
  return y #t,

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

def hamil(coords):
    
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

def getgrad(coords):
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

def dynamics_fn(t, coords):
    # print("Here")
    dcoords = getgrad(coords)
    dic1 = np.split(dcoords,2*input_dim1)
    S = np.concatenate([dic1[input_dim1]])
    for ii in np.arange(input_dim1+1,2*input_dim1,1):
        S = np.concatenate([S, dic1[ii]])
    for ii in np.arange(0,input_dim1,1):
        S = np.concatenate([S, -dic1[ii]])
    return S

chains = 1
y0 = np.zeros(args.input_dim)
N = 100
L = 25
steps = L*40
t_span = [0,L]
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-10}
burn = 1000

hmc_fin = np.zeros((chains,N,int(args.input_dim/2)))
hmc_accept = np.zeros((chains,N))
for ss in np.arange(0,chains,1):
    rk_req = np.zeros((N,int(args.input_dim/2)))
    rk_accept = np.zeros(N)
    RK = np.zeros((args.input_dim,steps,N))
    for ii in np.arange(0,int(args.input_dim/2),1):
        y0[ii] = 0.0 # 0.01
    for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
        y0[ii] = norm(loc=0,scale=1).rvs() # -3.0 # -0.87658921 # 
    for ii in np.arange(0,N,1):#
        # y0 = HNN_sto[:,0,ii]
        # y0 = hnn_fin[ss,ii,:].reshape(int(args.input_dim/2))
        data = leapfrog ( dynamics_fn, t_span, y0, steps, int(args.input_dim)) # get_dataset(y0=y0, samples=1, test_split=1.0)
        RK[:,:,ii] = data[:,0:steps]
        yhamil1 = np.zeros(args.input_dim)
        for jj in np.arange(0,args.input_dim,1):
            yhamil1[jj] = data[jj,steps-1] # data.get("coords")[steps-1,jj]
        H_star = hamil(yhamil1) # func1(yhamil1) # 
        H_prev = hamil(y0) # func1(y0) # 
        alpha = np.minimum(1,np.exp(H_prev - H_star))
        if alpha > uniform().rvs():
            y0[0:int(args.input_dim/2)] = data[0:int(args.input_dim/2),steps-1] # data.get("coords")[steps-1,0:int(args.input_dim/2)]
            rk_req[ii,0:int(args.input_dim/2)] = data[0:int(args.input_dim/2),steps-1] # data.get("coords")[steps-1,0:int(args.input_dim/2)]
            rk_accept[ii] = 1
        else:
            rk_req[ii+1,:] = y0[0:int(args.input_dim/2)]
        for jj in np.arange(int(args.input_dim/2),args.input_dim,1):
            y0[jj] = norm(loc=0,scale=1).rvs()
        print("Sample: "+str(ii)+" Chain: "+str(ss))
        # y0[0:int(args.input_dim/2)] = 0.0
        # y0[int(args.input_dim/2):int(args.input_dim)] = mome[ii+1]
    hmc_accept[ss,:] = rk_accept
    hmc_fin[ss,:,:] = rk_req

ess_hmc = np.zeros((chains,int(args.input_dim/2)))
for ss in np.arange(0,chains,1):
    hmc_tf = tf.convert_to_tensor(hmc_fin[ss,burn:N,:])
    ess_hmc[ss,:] = np.array(tfp.mcmc.effective_sample_size(hmc_tf))