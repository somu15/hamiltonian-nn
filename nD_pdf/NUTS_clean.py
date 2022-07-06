#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:01:00 2022

@author: dhulls
"""

# import numpy as np
from numpy import log, exp, sqrt
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
import csv
import arviz as az

EXPERIMENT_DIR = './nD_pdf'
sys.path.append(EXPERIMENT_DIR)

import random
from data import get_dataset, get_field, get_trajectory, dynamics_fn, hamiltonian_fn
from nn_models import MLP
from hnn import HNN
from utils import L2_loss
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
    return {'input_dim': 4,
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

# def get_model(args, baseline):
#     output_dim = args.input_dim if baseline else args.input_dim
#     nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
#     model = HNN(args.input_dim, differentiable_model=nn_model,
#               field_type=args.field_type, baseline=baseline)

#     model_name = 'baseline' if baseline else 'hnn'
#     # path = "{}/ndpdf{}-{}.tar".format(args.save_dir, RK4, model_name) #
#     path = "ndpdf-hnn.tar" # .format(args.save_dir, RK4, model_name) # ndpdf-hnn.tar
#     model.load_state_dict(torch.load(path))
#     return model

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
  return y

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

def compute_slice(h_val):
    uni1 = uniform(loc=0,scale=np.exp(-h_val)).rvs()
    return np.log(uni1)

# def find_reasonable_epsilon(y0):
#     """ Heuristic for choosing an initial value of epsilon """
#     epsilon = 1.
#     k = 1.
#     t_span1 = [0, epsilon]
#     hnn_ivp1 = leapfrog ( dynamics_fn, t_span1, y0, 1, int(args.input_dim))
#     epsilon = 0.5 * k * epsilon
#     yhamil = hnn_ivp1[:,1]
#     H_star = hamil(yhamil)
#     H_prev = hamil(y0)
#     logacceptprob = H_prev - H_star
#     a = 1. if logacceptprob > np.log(0.5) else -1.
#     while a * logacceptprob > -a * np.log(2):
#         epsilon = epsilon * (2. ** a)
#         t_span1 = [0, epsilon]
#         hnn_ivp1 = leapfrog ( dynamics_fn, t_span1, y0, 1, int(args.input_dim))
#         yhamil = hnn_ivp1[:,1]
#         H_star = hamil(yhamil)
#         logacceptprob = H_prev - H_star

#     print("find_reasonable_epsilon=", epsilon)

#     return epsilon


def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)


def build_tree(theta, r, logu, v, j, epsilon, joint0):
    """The main recursion."""
    if (j == 0):
        # joint0 = hamil(hnn_ivp1[:,1])
        t_span1 = [0,v * epsilon]
        y1 = np.concatenate((theta, r), axis=0)
        hnn_ivp1 = leapfrog ( dynamics_fn, t_span1, y1, 1, int(args.input_dim)) # integrate_model(hnn_model, t_span1, y1, 1, **kwargs1)
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
M = 25000 # 125000
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
    y0[ii] = norm(loc=0,scale=1).rvs() #  theta4[ii-100] #  3.0 # -0.87658921 #

# Choose a reasonable first epsilon by a simple heuristic.
# epsilon = find_reasonable_epsilon(y0)

# Parameters to the dual averaging algorithm.
epsilon = 0.025 # 0.025
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
H_store = np.zeros(M)

for m in np.arange(0, M + Madapt, 1):
    print(m)
    for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
        y0[ii] = norm(loc=0,scale=1).rvs() #  3.0 # -0.87658921 #
    # Resample momenta.
    # r0 = np.random.normal(0, 1, D)
    # r_sto = y0[int(args.input_dim/2):int(args.input_dim)]

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
        
    traj_len[m] = j

    alpha1 = np.minimum(1,np.exp(joint - hamil(np.concatenate((samples[m, :], r_sto), axis=0))))
    if alpha1 > uniform().rvs():
        y0[0:int(args.input_dim/2)] = samples[m, :]
        H_store[m] = hamil(np.concatenate((samples[m, :], r_sto), axis=0))
    else:
        samples[m, :] = samples[m-1, :]
        HNN_accept[m] = 0
        H_store[m] = joint

burn = 5000

ess_hnn = np.zeros((chains,int(args.input_dim/2)))
for ss in np.arange(0,chains,1):
    hnn_tf = tf.convert_to_tensor(samples[burn:M,:])
    ess_hnn[ss,:] = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
    