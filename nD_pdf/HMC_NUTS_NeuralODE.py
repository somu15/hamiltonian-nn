#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:10:10 2022

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
    s = torch.from_numpy(np.array([363, 352, 631, 267, 372, 768, 593, 330, 974, 697, 922, 286, 906,
           238, 987, 778, 971, 972, 913, 727])) # np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False)
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

######## NUTS code ########

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

def compute_slice(h_val):
    uni1 = uniform(loc=0,scale=np.exp(-h_val)).rvs()
    return np.log(uni1)

def find_reasonable_epsilon(y0):
    """ Heuristic for choosing an initial value of epsilon """
    epsilon = 1.
    k = 1.
    t_span1 = [0, epsilon]
    hnn_ivp1 = leapfrog ( dynamics_fn, t_span1, y0, 1, int(args.input_dim))
    epsilon = 0.5 * k * epsilon
    yhamil = hnn_ivp1[:,1]
    H_star = hamil(yhamil)
    H_prev = hamil(y0)
    logacceptprob = H_prev - H_star
    a = 1. if logacceptprob > np.log(0.5) else -1.
    while a * logacceptprob > -a * np.log(2):
        epsilon = epsilon * (2. ** a)
        t_span1 = [0, epsilon]
        hnn_ivp1 = leapfrog ( dynamics_fn, t_span1, y0, 1, int(args.input_dim))
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
        y1 = np.concatenate((theta, r), axis=0)
        hnn_ivp1 = leapfrog ( dynamics_fn, t_span1, y1, 1, int(args.input_dim)) # integrate_model(hnn_model, t_span1, y1, 1, **kwargs1)
        thetaprime = hnn_ivp1[0:int(args.input_dim/2), 1].reshape(int(args.input_dim/2))
        rprime = hnn_ivp1[int(args.input_dim/2):int(args.input_dim), 1].reshape(int(args.input_dim/2))
        joint = hamil(hnn_ivp1[:,1])
        nprime = int(logu <= np.exp(-joint))
        sprime = int((np.log(logu) + joint) <= 1000.)
        monitor = np.log(logu) + joint
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        # alphaprime = min(1., np.exp(joint - joint0))
        alphaprime = min(1., np.exp(joint0 - joint))
        # print(np.exp(joint0 - joint))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor = build_tree(theta, r, logu, v, j - 1, epsilon, joint0)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, _, _, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor = build_tree(thetaminus, rminus, logu, v, j - 1, epsilon, joint0)
            else:
                _, _, thetaplus, rplus, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor = build_tree(thetaplus, rplus, logu, v, j - 1, epsilon, joint0)
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

    return thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor


D = int(args.input_dim/2)
M = 10000 # 125000 # 125000
Madapt = 0 # 500
theta0 = np.ones(D) # np.array([-2.92564691,  8.49055319, 71.92149821]) #  np.random.normal(0, 1, D)
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
mu = np.log(10. * epsilon)

# Initialize dual averaging algorithm.
epsilonbar = 1
chains = 1
Hbar = 0


HNN_accept = np.ones(M)
traj_len = np.zeros(M)
H_store = np.zeros(M)

monitor_err = np.zeros(M)

for m in np.arange(1926, M + Madapt, 1):
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
    # logu = compute_slice(joint)
    logu = np.random.uniform(0, np.exp(-joint))

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
            thetaminus, rminus, _, _, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor = build_tree(thetaminus, rminus, logu, v, j, epsilon, joint)
        else:
            _, _, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor = build_tree(thetaplus, rplus, logu, v, j, epsilon, joint)

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
        monitor_err[m] = monitor
        
    traj_len[m] = j
    y0[0:int(args.input_dim/2)] = samples[m, :]
    H_store[m] = hamil(np.concatenate((samples[m, :], r_sto), axis=0))
    

    # alpha1 = 1.0 # np.minimum(1,np.exp(joint - hamil(np.concatenate((samples[m, :], r_sto), axis=0))))
    # if alpha1 > uniform().rvs():
    #     y0[0:int(args.input_dim/2)] = samples[m, :]
    #     H_store[m] = hamil(np.concatenate((samples[m, :], r_sto), axis=0))
    # else:
    #     samples[m, :] = samples[m-1, :]
    #     HNN_accept[m] = 0
    #     H_store[m] = joint

py = np.zeros((1000,2))
for ii in np.arange(500,1925,1):
    p1 = samples[ii,:]
    init_params = convert_to_nn_tensor(p1, 5)
    func = ODEFuncGrad(init_params).to(device)
    pred_y = odeint(func.float(), true_y0, t).to(device)
    # py = py + pred_y.detach().numpy().reshape(1000,2)
    print(ii)
    # plt.plot(pred_y.detach().numpy()[:,:,0], pred_y.detach().numpy()[:,:,1])

init_params = convert_to_nn_tensor(np.mean(samples[500:1925],axis=0), 5)
func = ODEFuncGrad(init_params).to(device)
pred_y = odeint(func.float(), true_y0, t).to(device)
py = pred_y.detach().numpy()

burn = 5000

ess_hnn = np.zeros((chains,int(args.input_dim/2)))
for ss in np.arange(0,chains,1):
    hnn_tf = tf.convert_to_tensor(samples[burn:M,:])
    ess_hnn[ss,:] = np.array(tfp.mcmc.effective_sample_size(hnn_tf))