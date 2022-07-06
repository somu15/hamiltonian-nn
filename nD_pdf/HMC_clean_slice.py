#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 00:08:20 2022

@author: dhulls
"""

input_dim1 = 2

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

## Ref: 50; 2 and 200



def get_args():
    return {'input_dim': 4,
         'hidden_dim': 100,
         'learn_rate': 5e-4,
         'nonlinearity': 'sine',
         'total_steps': 25000,
         'field_type': 'solenoidal',
         'print_every': 200,
         'name': 'ndpdf',
         'use_rk4' : 'True',
         'gridsize': 10,
         'input_noise': 0.01,
         'seed': 0,
         'save_dir': './{}'.format(EXPERIMENT_DIR),
         'fig_dir': './figures'}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())


# np.random.seed(args.seed)
# R = 2.5
# field = get_field(xmin=-R, xmax=R, ymin=-R, ymax=R, gridsize=15)
# data = get_dataset()

# # plot config
# fig = plt.figure(figsize=(3, 3), facecolor='white', dpi=DPI)

# x, y, dx, dy, t = get_trajectory(radius=2.4, y0=np.array([2,0]), noise_std=0)
# plt.scatter(x,y,c=t,s=14, label='data')
# plt.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
#            cmap='gray_r', color=(.5,.5,.5))
# plt.xlabel("$q$", fontsize=14)
# plt.ylabel("p", rotation=0, fontsize=14)
# plt.title("Dynamics")
# plt.legend(loc='upper right')

# plt.tight_layout() ; plt.show()

# def get_model(args, baseline):
#     output_dim = args.input_dim if baseline else args.input_dim
#     nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
#     model = HNN(args.input_dim, differentiable_model=nn_model,
#               field_type=args.field_type, baseline=baseline)

#     model_name = 'baseline' if baseline else 'hnn'
#     # path = "{}/ndpdf{}-{}.tar".format(args.save_dir, RK4, model_name) # 
#     path = "1D_Gauss_Mix_demo_035.tar" # .format(args.save_dir, RK4, model_name) # 
#     model.load_state_dict(torch.load(path))
#     return model

# def get_vector_field(model, **kwargs):
#     field = get_field(**kwargs)
#     np_mesh_x = field['x']

#     # run model
#     mesh_x = torch.tensor( np_mesh_x, requires_grad=True, dtype=torch.float32)
#     mesh_dx = model.time_derivative(mesh_x)
#     return mesh_dx.data.numpy()

# def comp_factor(t):
#     if t!=0. and t!=None:
#         factor = (1-np.exp(-t))/t
#     else:
#         factor = 0.
#     return factor

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
      anew   = dydt ( t, y[:,i] ) # *comp_factor(t[i])
    else:
      t[i]   = t[i-1] + dt
      aold   = anew
      for j in range ( 0, int(dim/2) ):
          y[j,i] = y[j,i-1] + dt * ( y[(j+int(dim/2)),i-1] + 0.5 * dt * aold[(j+int(dim/2))] )
      anew   = dydt ( t, y[:,i] ) # *comp_factor(t[i])
      for j in range ( 0, int(dim/2) ):
          y[(j+int(dim/2)),i] = y[(j+int(dim/2)),i-1] + 0.5 * dt * ( aold[(j+int(dim/2))] + anew[(j+int(dim/2))] )
  return y #t,

# def integrate_model(model, t_span, y0, n, **kwargs):

#     def fun(t, np_x):
#         x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,args.input_dim)
#         dx = model.time_derivative(x).data.numpy().reshape(-1)
#         return dx

#     # return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
#     return leapfrog(fun, t_span, y0, n, args.input_dim)

# # base_model = get_model(args, baseline=False)
# hnn_model = get_model(args, baseline=False)

def compute_slice(h_val):
    uni1 = uniform(loc=0,scale=np.exp(h_val)).rvs()
    return (uni1)

def hamil(coords):
    
    #******** 2D Gaussian Four Mixtures #********
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

    return H

def dynamics_fn(t, coords):
    # print("Here")
    dcoords = autograd.grad(hamil)(coords) # func1
    dic1 = np.split(dcoords,2*input_dim1)
    S = np.concatenate([dic1[input_dim1]])
    for ii in np.arange(input_dim1+1,2*input_dim1,1):
        S = np.concatenate([S, dic1[ii]])
    for ii in np.arange(0,input_dim1,1):
        S = np.concatenate([S, -dic1[ii]])
    return S

chains = 1
y0 = np.zeros(args.input_dim)
N = 1000
L = 5
steps = L*1 #  20
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
        logu = float(-H_prev - np.random.exponential(1, size=1)) # compute_slice(H_prev)
        alpha = int(logu <= (-H_star))
        if alpha:
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