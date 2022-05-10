#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 22:22:02 2022

@author: dhulls
"""

# import numpy as np
from numpy import log, exp, sqrt
import torch, time, sys
# import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from statsmodels.distributions.empirical_distribution import ECDF
import tensorflow as tf
import tensorflow_probability as tfp
# import csv
# import arviz as az

EXPERIMENT_DIR = './nD_pdf'
sys.path.append(EXPERIMENT_DIR)

# import random
# from data import get_dataset, get_field, get_trajectory, dynamics_fn, hamiltonian_fn
from nn_models import MLP
from hnn import HNN
# from utils import L2_loss
from scipy.stats import norm
from scipy.stats import uniform
import pandas as pd
from pandas.plotting import scatter_matrix

# from .helpers import progress_range

# __all__ = ['nuts6']

DPI = 300
FORMAT = 'pdf'
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2
RK4 = ''

## Ref: 50; 2 and 200



def get_args():
    return {'input_dim': 6,
         'hidden_dim': 100,
         'learn_rate': 5e-4,
         'nonlinearity': 'sine',
         'total_steps': 100000,
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

def get_model(args, baseline):
    output_dim = args.input_dim if baseline else args.input_dim
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=baseline)

    model_name = 'baseline' if baseline else 'hnn'
    # path = "{}/ndpdf{}-{}.tar".format(args.save_dir, RK4, model_name) #
    path = "ndpdf-hnn.tar" # .format(args.save_dir, RK4, model_name) # ndpdf-hnn
    model.load_state_dict(torch.load(path))
    return model

# def leapfrog(dydt, tspan, y0, n, dim):
#     t0 = tspan[0]
#     tstop = tspan[1]
#     dt = ( tstop - t0 ) / n
#     aold = dydt ( t0, y0 )
#     # make half step in r
#     rprime = y0[int(dim/2) : int(dim)] + 0.5 * dt * aold[int(dim/2) : int(dim)]
#     # make new step in theta
#     thetaprime = y0[0 : int(dim/2)] + dt * rprime
#     #compute new gradient
#     anew = dydt ( tstop, np.concatenate((thetaprime, rprime), axis=0) )
#     # make half step in r again
#     rprime = rprime + 0.5 * dt * anew[int(dim/2) : int(dim)]
    
#     yret = np.zeros((dim,2))
#     yret[:,1] = np.concatenate((thetaprime, rprime), axis=0)
    
#     return yret

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

def integrate_model(model, t_span, y0, n, **kwargs):

    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,args.input_dim)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx
    return leapfrog(fun, t_span, y0, n, args.input_dim)

hnn_model = get_model(args, baseline=False)

def hamil(coords):
    
    # ******** 20D German Credit Data #******** (200 Neurons, 100000 stepa)
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

    #******** 5D Gaussian #********
    # dic1 = np.split(coords,args.input_dim)
    # var1 = np.array([1.,1.,1.,1.,1.])
    # term1 = dic1[0]**2/(2*var1[0])
    # for ii in np.arange(1,5,1):
    #     term1 = term1 + dic1[ii]**2/(2*var1[ii])
    # term2 = dic1[5]**2/2
    # for ii in np.arange(6,10,1):
    #     term2 = term2 + dic1[ii]**2/2
    # H = term1 + term2

    #******** 2D Example from Rashki MSSP #********
    # dic1 = np.split(coords,args.input_dim)
    # # tmp1 = ((30 / (4 * (dic1[0] + 2)**2) / 9) + (dic1[1]**2 / 25)**2 + 1)
    # # tmp2 = (20 / (((dic1[0] - 2.5)**2 / 4) + ((dic1[1] - 0.5)**2 / 25)**2 + 1) - 5)
    # # term1 = tmp1 + tmp2
    # tmp1 = (4 - dic1[0]) * (dic1[0] > 3.5) + (0.85 - 0.1 * dic1[0]) * (dic1[0] <= 3.5)
    # tmp2 = (4 - dic1[1]) * (dic1[1] > 3.5) + (0.85 - 0.1 * dic1[1]) * (dic1[1] <= 3.5)
    # term1 = tmp1 * (tmp1 < tmp2) + tmp2 * (tmp2 < tmp1)
    # term2 = dic1[2]**2/2 + dic1[3]**2/2
    # H = term1 + term2

    #******** 5D Ill-Conditioned Gaussian #********
    # dic1 = np.split(coords,args.input_dim)
    # var1 = np.array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])
    # term1 = dic1[0]**2/(2*var1[0])
    # for ii in np.arange(1,5,1):
    #     term1 = term1 + dic1[ii]**2/(2*var1[ii])
    # term2 = dic1[5]**2/2
    # for ii in np.arange(6,10,1):
    #     term2 = term2 + dic1[ii]**2/2
    # H = term1 + term2

    #******** 2D Funnel #********
    # dic1 = np.split(coords,args.input_dim)
    # term1 = dic1[0]**2/(2*3**2)
    # for ii in np.arange(1,2,1):
    #     term1 = term1 + dic1[ii]**2/(2 * (2.718281828459045**(dic1[0] / 2))**2)
    # term2 = 0.0
    # for ii in np.arange(2,4,1):
    #     term2 = term2 + dic1[ii]**2/2 # term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
    # H = term1 + term2

    #******** nD Funnel #********
    # dic1 = np.split(coords,args.input_dim)
    # term1 = dic1[0]**2/(2*3**2)
    # for ii in np.arange(1,int(args.input_dim/2),1):
    #     term1 = term1 + dic1[ii]**2/(2 * (2.718281828459045**(dic1[0] / 2))**2)
    # term2 = 0.0
    # for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
    #     term2 = term2 + dic1[ii]**2/2 # term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
    # H = term1 + term2
    
    # ********* nD Heirarchical (https://crackedbassoon.com/writing/funneling) *********
    # dic1 = np.split(coords,args.input_dim)
    # term1 = dic1[0]**2/2
    # term1 = term1 - np.log(2 / (np.pi * (1 + np.exp(dic1[1])**2)))
    # for ii in np.arange(2,int(args.input_dim/2),1):
    #     term1 = term1 + (dic1[ii] - dic1[0])**2 / (2 * np.exp(dic1[1])**2)
    # term2 = 0.0
    # for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
    #     term2 = term2 + dic1[ii]**2/2 # term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
    # H = term1 + term2

    #******** 2D Rosenbrock #********
    # dic1 = np.split(coords,args.input_dim)
    # a = 1
    # b = 100
    # p = 20
    # term1 = (b*(dic1[1]-dic1[0]**2)**2+(a-dic1[0])**2)/p
    # term2 = 1*dic1[2]**2/2+1*dic1[3]**2/2
    # H = term1 + term2

    #******** 3D Rosenbrock #********
    # dic1 = np.split(coords,args.input_dim)
    # term1 = (100 * (dic1[1] - dic1[0]**2)**2 + (1 - dic1[0])**2 + 100 * (dic1[2] - dic1[1]**2)**2 + (1 - dic1[1]**2)) / 20
    # term2 = 1*dic1[3]**2/2+1*dic1[4]**2/2+1*dic1[5]**2/2
    # H = term1 + term2

    #******** nD Rosenbrock #********
    dic1 = np.split(coords,args.input_dim)
    term1 = 0.0
    for ii in np.arange(0,int(args.input_dim/2)-1,1):
        term1 = term1 + (100 * (dic1[ii+1] - dic1[ii]**2)**2 + (1 - dic1[ii])**2) / 20.0
    term2 = 0.0
    for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
        term2 = term2 + 1*dic1[ii]**2/2
    H = term1 + term2
    
    # ******** 100D Gaussian by Radford Neal #********
    # dic1 = np.split(coords,args.input_dim)
    # var1 = np.arange(0.01,1.01,0.01)
    # term1 = 0.0
    # for ii in np.arange(0,int(args.input_dim/2),1):
    #     term1 = term1 + dic1[ii]**2/(2*var1[ii]**2)
    # term2 = 0.0
    # for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
    #     term2 = term2 + dic1[ii]**2/2
    # H = term1 + term2
    
    #******** 100D Allen-Cahn #********
    # dic1 = np.split(coords,args.input_dim)
    # term1 = 0.0
    # h = 1/(args.input_dim/2)
    # for ii in np.arange(0,int(args.input_dim/2)-1,1):
    #     tmp1 = (1-dic1[ii+1]**2)**2
    #     tmp2 =  (1-dic1[ii]**2)**2
    #     term1 = term1 + 1/(2*h) * (dic1[ii+1] - dic1[ii])**2 + h/2 * (tmp1 + tmp2)
    #     # tmp1 = dic1[ii+1] + dic1[ii]
    #     # term1 = term1 + 1/(2*h) * (dic1[ii+1] - dic1[ii])**2 + h/2 * (1 - tmp1**2)**2
    # term2 = 0.0
    # for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
    #     term2 = term2 + 1*dic1[ii]**2/2
    # H = term1 + term2

    #******** nD Even Rosenbrock #********
    # dic1 = np.split(coords,args.input_dim)
    # input_dim1 = args.input_dim/2
    # term1 = 0.0
    # for ii in np.arange(0,int(input_dim1/2),1):
    #     ind1 = ii
    #     ind2 = ii+1
    #     term1 = term1 + ((dic1[ind1] - 1.0)**2 - 100.0 * (dic1[ind2] - dic1[ind1]**2)**2) / 20.0 # (100 * (dic1[ii+1] - dic1[ii]**2)**2 + (1 - dic1[ii])**2) / 20.0
    # term2 = 0.0
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + 1*dic1[ii]**2/2
    # H = term1 + term2

    #******** 1D Gaussian Mixture #********
    # q, p = np.split(coords,2)
    # mu1 = 1.0
    # mu2 = -1.0
    # sigma = 0.35
    # term1 = -np.log(0.5*(np.exp(-(q-mu1)**2/(2*sigma**2)))+0.5*(np.exp(-(q-mu2)**2/(2*sigma**2))))
    # H = term1 + p**2/2 # Normal PDF

    #******** 2D Gaussian Four Mixtures #********
    # q1, q2, p1, p2 = np.split(coords,4)
    # sigma_inv = np.array([[1.,0.],[0.,1.]])
    # term1 = 0.

    # mu = np.array([3.,0.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])

    # mu = np.array([-3.,0.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])

    # mu = np.array([0.,3.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])

    # mu = np.array([0.,-3.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])

    # term1 = -np.log(term1)
    # term2 = p1**2/2+p2**2/2
    # H = term1 + term2

    #******** 2D Highly Correlated Gaussian #********
    # q1, q2, p1, p2 = np.split(coords,4)
    # sigma_inv = np.array([[50.25125628,-24.87437186],[-24.87437186,12.56281407]])
    # term1 = 0.

    # mu = np.array([0.,0.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])

    # term1 = -np.log(term1)
    # term2 = p1**2/2+p2**2/2
    # H = term1 + term2

    return H

def compute_slice(h_val):
    uni1 = uniform(loc=0,scale=np.exp(-h_val)).rvs()
    return np.log(uni1)

def find_reasonable_epsilon(y0):
    """ Heuristic for choosing an initial value of epsilon """
    epsilon = 1.
    k = 1.
    t_span1 = [0, epsilon]
    kwargs1 = {'t_eval': np.linspace(t_span1[0], t_span1[1], 1), 'rtol': 1e-10}
    hnn_ivp1 = integrate_model(hnn_model, t_span1, y0, 1, **kwargs1)
    epsilon = 0.5 * k * epsilon
    yhamil = hnn_ivp1[:,1]
    H_star = hamil(yhamil)
    H_prev = hamil(y0)
    logacceptprob = H_prev - H_star
    a = 1. if logacceptprob > np.log(0.5) else -1.
    while a * logacceptprob > -a * np.log(2):
        epsilon = epsilon * (2. ** a)
        t_span1 = [0, epsilon]
        kwargs1 = {'t_eval': np.linspace(t_span1[0], t_span1[1], 1), 'rtol': 1e-10}
        hnn_ivp1 = integrate_model(hnn_model, t_span1, y0, 1, **kwargs1)
        yhamil = hnn_ivp1[:,1]
        H_star = hamil(yhamil)
        logacceptprob = H_prev - H_star

    print("find_reasonable_epsilon=", epsilon)

    return epsilon


def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)


def build_tree(theta, r, logu, v, j, epsilon, joint0):
    """The main recursion."""
    if (j == 0):
        # joint0 = hamil(hnn_ivp1[:,1])
        t_span1 = [0,v * epsilon]
        kwargs1 = {'t_eval': np.linspace(t_span1[0], t_span1[1], 1), 'rtol': 1e-10}
        y1 = np.concatenate((theta, r), axis=0)
        hnn_ivp1 = integrate_model(hnn_model, t_span1, y1, 1, **kwargs1)
        thetaprime = hnn_ivp1[0:int(args.input_dim/2), 1].reshape(int(args.input_dim/2))
        rprime = hnn_ivp1[int(args.input_dim/2):int(args.input_dim), 1].reshape(int(args.input_dim/2))
        joint = hamil(hnn_ivp1[:,1])
        nprime = int(logu < joint)
        sprime = int((logu - 1000.) < joint)
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        # alphaprime = min(1., np.exp(joint - joint0))
        alphaprime = min(1., np.exp(joint0 - joint))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime = build_tree(theta, r, logu, v, j - 1, epsilon, joint0)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, _, _, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(thetaminus, rminus, logu, v, j - 1, epsilon, joint0)
            else:
                _, _, thetaplus, rplus, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(thetaplus, rplus, logu, v, j - 1, epsilon, joint0)
            # Choose which subtree to propagate a sample up from.
            if (np.random.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                thetaprime = thetaprime2[:]
                rprime = rprime2[:]
            # Update the number of valid points.
            nprime = int(nprime) + int(nprime2)
            # Update the stopping criterion.
            sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus))
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime

D = int(args.input_dim/2)
M = 25000
Madapt = 0 # 500
theta0 = np.ones(D) # np.random.normal(0, 1, D)
delta = 0.2
D = len(theta0)
samples = np.empty((M + Madapt, D), dtype=float)
samples[0, :] = theta0

y0 = np.zeros(args.input_dim)
for ii in np.arange(0,int(args.input_dim/2),1):
    y0[ii] = theta0[ii]
for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
    y0[ii] = norm(loc=0,scale=1).rvs() #  3.0 # -0.87658921 #

# Choose a reasonable first epsilon by a simple heuristic.
# epsilon = find_reasonable_epsilon(y0)

# Parameters to the dual averaging algorithm.
epsilon = 0.025 # 0.005
gamma = 0.05
t0 = 10
kappa = 0.75
mu = log(10. * epsilon)

# Initialize dual averaging algorithm.
epsilonbar = 1
chains = 1
Hbar = 0


HNN_accept = np.ones(M)
traj_len = np.zeros(M)

for m in np.arange(0, M + Madapt, 1):
    print(m)
    for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
        y0[ii] = norm(loc=0,scale=1).rvs() #  3.0 # -0.87658921 #
    # Resample momenta.
    # r0 = np.random.normal(0, 1, D)

    #joint lnp of theta and momentum r
    joint = hamil(y0) # logp - 0.5 * np.dot(r0, r0.T)

    # Resample u ~ uniform([0, exp(joint)]).
    # Equivalent to (log(u) - joint) ~ exponential(1).
    logu = compute_slice(joint)

    # if all fails, the next sample will be the previous one
    samples[m, :] = samples[m - 1, :]
    # lnprob[m] = lnprob[m - 1]

    # initialize the tree
    thetaminus = samples[m - 1, :]
    thetaplus = samples[m - 1, :]
    rminus = y0[int(args.input_dim/2):int(args.input_dim)]
    rplus = y0[int(args.input_dim/2):int(args.input_dim)]
    # gradminus = grad[:]
    # gradplus = grad[:]

    j = 0  # initial heigth j = 0
    n = 1  # Initially the only valid point is the initial point.
    s = 1  # Main loop: will keep going until s == 0.

    while (s == 1):
        # Choose a direction. -1 = backwards, 1 = forwards.
        v = int(2 * (np.random.uniform() < 0.5) - 1)

        # Double the size of the tree.
        if (v == -1):
            thetaminus, rminus, _, _, thetaprime, rprime, nprime, sprime, alpha, nalpha = build_tree(thetaminus, rminus, logu, v, j, epsilon, joint)
        else:
            _, _, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alpha, nalpha = build_tree(thetaplus, rplus, logu, v, j, epsilon, joint)

        # Use Metropolis-Hastings to decide whether or not to move to a
        # point from the half-tree we just generated.
        _tmp = min(1, float(nprime) / float(n))
        if (sprime == 1) and (np.random.uniform() < _tmp):
            samples[m, :] = thetaprime[:]
            r_sto = rprime
        # Update number of valid points we've seen.
        n += nprime
        # Decide if it's time to stop.
        s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
        # Increment depth.
        j += 1

    # Do adaptation of epsilon if we're still doing burn-in.
    # eta = 1. / float(m + t0)
    # Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
    # epsilon = exp(mu - sqrt(m) / gamma * Hbar)
    # eta = m ** -kappa
    # epsilonbar = exp((1. - eta) * log(epsilonbar) + eta * log(epsilon))
    # if (m <= Madapt):
    #     epsilon = exp(mu - sqrt(m) / gamma * Hbar)
    #     eta = m ** -kappa
    #     epsilonbar = exp((1. - eta) * log(epsilonbar) + eta * log(epsilon))
    # else:
    #     epsilon = epsilonbar
    traj_len[m] = j
    alpha1 =  np.minimum(1,np.exp(joint - hamil(np.concatenate((samples[m, :], r_sto), axis=0))))
    # alpha1 = alpha / float(nalpha)
    if alpha1 > uniform().rvs():
        y0[0:int(args.input_dim/2)] = samples[m, :]
    else:
        samples[m, :] = samples[m-1, :]
        HNN_accept[m] = 0
# samples = samples[Madapt:, :]
    # lnprob = lnprob[Madapt:]

burn = 5000

ess_hnn = np.zeros((chains,int(args.input_dim/2)))
for ss in np.arange(0,chains,1):
    hnn_tf = tf.convert_to_tensor(samples[burn:M,:])
    ess_hnn[ss,:] = np.array(tfp.mcmc.effective_sample_size(hnn_tf))

fig = plt.figure(figsize=(6, 6))
plt.plot(samples[burn:M, 1], samples[burn:M, 2], 'b+')
# plt.xticks(np.array([]))
# plt.yticks(np.array([]))

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
ax.scatter(samples[burn:M,0],samples[burn:M,1],samples[burn:M,2],s=2)
# ax.scatter(samples_ref[burn:M,0],samples_ref[burn:M,1],samples_ref[burn:M,2],s=2)

plt.hist(samples[burn:M,2],bins=50,density=True)
plt.hist(samples_ref[burn:M,2],bins=50,density=True)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot()
e_hnn1 = ECDF(samples[burn:M,2])
# e_hnn3 = ECDF(samples3[burn:M,2])
# e_hnn4 = ECDF(samples4[burn:M,2])
e_hmc = ECDF(samples_ref[burn:M,2])
plt.plot(e_hnn1.x,e_hnn1.y,label='L 100',linestyle='--',linewidth=1.8)
# plt.plot(e_hnn3.x,e_hnn3.y,label='L 250',linestyle='-.',linewidth=1.8)
# plt.plot(e_hnn4.x,e_hnn4.y,label='L 400',linestyle=':',linewidth=1.8)
plt.plot(e_hmc.x,e_hmc.y,label='Reference',linewidth=3)
plt.legend()

df1 = pd.DataFrame(samples[burn:M,10:20], columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']) # 
scatter_matrix(df1, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
# plt.savefig('/Users/dhulls/Desktop/Logistic_2.pdf', format='pdf', bbox_inches = "tight")

# fig = plt.figure(figsize=(6, 6))
# plt.plot(X_p, np.exp(log_dens), linewidth = 5.5)
# plt.xticks(np.array([]))
# plt.yticks(np.array([]))

# def test_nuts6():
#    """ Example usage of nuts6: sampling a 2d highly correlated Gaussian distribution """

# class Counter:
#     def __init__(self, c=0):
#         self.c = c
#
# c = Counter()
# def correlated_normal(theta):
#     """
#     Example of a target distribution that could be sampled from using NUTS.
#     (Although of course you could sample from it more efficiently)
#     Doesn't include the normalizing constant.
#     """
#
#     # Precision matrix with covariance [1, 1.98; 1.98, 4].
#     # A = np.linalg.inv( cov )
#     A = np.asarray([[50.251256, -24.874372],
#                     [-24.874372, 12.562814]])
#
#     # add the counter to count how many times this function is called
#     c.c += 1
#
#     grad = -np.dot(theta, A)
#     logp = 0.5 * np.dot(grad, theta.T)
#     return logp, grad



# mean = np.zeros(2)
# cov = np.asarray([[1, 1.98],
#                   [1.98, 4]])

# # print('Running HMC with dual averaging and trajectory length %0.2f...' % delta)
# # samples, epsilon = nuts6(M, Madapt, theta0, delta)
# # print('Done. Final epsilon = %f.' % epsilon)
# # print('(M+Madapt) / Functions called: %f' % ((M+Madapt)/float(c.c)))

# # samples = samples[1::10, :]
# # print('Percentiles')
# # print (np.percentile(samples, [16, 50, 84], axis=0))
# # print('Mean')
# # print (np.mean(samples, axis=0))
# # print('Stddev')
# # print (np.std(samples, axis=0))

# # try:
# #     import matplotlib.pyplot as plt
# # except ImportError:
# #     import pylab as plt
# temp = np.random.multivariate_normal(mean, cov, size=10000)
# # plt.subplot(1,3,1)
# plt.plot(temp[:, 0], temp[:, 1], '.')
# plt.plot(samples[1000:10000, 0], samples[1000:10000, 1], 'r+')

# plt.subplot(1,3,2)
# plt.hist(samples[:,0], bins=50)
# plt.xlabel("x-samples")

# plt.subplot(1,3,3)
# plt.hist(samples[:,1], bins=50)
# plt.xlabel("y-samples")
# plt.show()

# ess_nuts = np.zeros((1,2))
# nuts_tf = tf.convert_to_tensor(samples[0:1000,:])
# ess_nuts[0,:] = np.array(tfp.mcmc.effective_sample_size(nuts_tf))

# if __name__ == "__main__":
#     test_nuts6()

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot()
plt.hist(samples[0:17300,50],bins=50,density=True)
ax.set_ylabel('Probability')
ax.set_xlabel('u_50')
ax.set_ylim([0,0.5])

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot()
plt.plot(samples[1:21,:].T)
ax.set_ylabel('u value')
ax.set_xlabel('index')
