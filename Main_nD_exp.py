#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:51:52 2021

@author: dhulls
"""

import torch, time, sys
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from statsmodels.distributions.empirical_distribution import ECDF

EXPERIMENT_DIR = './nD_pdf_exp'
sys.path.append(EXPERIMENT_DIR)

# from data import get_dataset, get_field, get_trajectory, dynamics_fn, hamiltonian_fn
from nn_models import MLP
from hnn import HNN
from utils import L2_loss
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
    return {'input_dim': 8,
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

def get_model(args, baseline):
    output_dim = args.input_dim if baseline else args.input_dim
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=baseline)

    model_name = 'baseline' if baseline else 'hnn'
    path = "{}/ndpdf{}-{}.tar".format(args.save_dir, RK4, model_name)
    model.load_state_dict(torch.load(path))
    return model

# def get_vector_field(model, **kwargs):
#     field = get_field(**kwargs)
#     np_mesh_x = field['x']

#     # run model
#     mesh_x = torch.tensor( np_mesh_x, requires_grad=True, dtype=torch.float32)
#     mesh_dx = model.time_derivative(mesh_x)
#     return mesh_dx.data.numpy()

def comp_factor(t):
    if t!=0. and t!=None:
        factor = (1-np.exp(-t))/t
    else:
        factor = 0.
    return factor

# def leapfrog ( dydt, tspan, y0, n, dim ):
#   t0 = tspan[0]
#   tstop = tspan[1]
#   dt = ( tstop - t0 ) / n

#   t = np.zeros ( n + 1 )
#   y = np.zeros ( [dim, n + 1] )

#   for i in range ( 0, n + 1 ):

#     if ( i == 0 ):
#       t[0]   = t0
#       for j in range ( 0, dim ):
#           y[j,0] = y0[j]
#       anew   = dydt ( t, y[:,i] ) # *comp_factor(t[i])**8
#     else:
#       t[i]   = t[i-1] + dt
#       aold   = anew
#       for j in range ( 0, int(dim/2) ):
#           y[j,i] = y[j,i-1] + dt * ( y[(j+int(dim/2)),i-1] + 0.5 * dt * aold[(j+int(dim/2))] )
#       anew   = dydt ( t, y[:,i] ) # *comp_factor(t[i])**8
#       for j in range ( 0, int(dim/2) ):
#           y[(j+int(dim/2)),i] = y[(j+int(dim/2)),i-1] + 0.5 * dt * ( aold[(j+int(dim/2))] + anew[(j+int(dim/2))] )
#   return y #t,

def leapfrog ( dydt, tspan, y0, n, dim ):
  t0 = tspan[0]
  tstop = tspan[1]
  dt = ( tstop - t0 ) / n
  # dt = dt*comp_factor(dt)

  t = np.zeros ( n + 1 )
  y = np.zeros ( [dim, n + 1] )

  for i in range ( 0, n + 1 ):

    if ( i == 0 ):
      t[0]   = t0
      for j in range ( 0, dim ):
          y[j,0] = y0[j]
      anew   = dydt ( t, y[:,i] ) # *comp_factor(t[i])**8
    else:
      t[i]   = t[i-1] + dt
      aold   = anew
      for j in range ( 0, int(dim/2) ):
          y[j,i] = y[j,i-1] + dt**2 * comp_factor(dt)**2 * ( y[(j+int(dim/2)),i-1] + 0.5 * dt**2 * comp_factor(dt)**2 * aold[(j+int(dim/2))] )
      anew   = dydt ( t, y[:,i] ) # *comp_factor(t[i])**8
      for j in range ( 0, int(dim/2) ):
          y[(j+int(dim/2)),i] = y[(j+int(dim/2)),i-1] + 0.5 * dt**2 * comp_factor(dt)**2 * ( aold[(j+int(dim/2))] + anew[(j+int(dim/2))] )
  return y #t,

def integrate_model(model, t_span, y0, n, **kwargs):

    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,args.input_dim)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx

    # return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
    return leapfrog(fun, t_span, y0, n, args.input_dim)

# base_model = get_model(args, baseline=False)
hnn_model = get_model(args, baseline=False)

def hamil(coords):

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
    # term2 = dic1[2]**2/2
    # for ii in np.arange(3,4,1):
    #     term2 = term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
    # H = term1 + term2

    #******** nD Funnel #********
    # dic1 = np.split(coords,args.input_dim)
    # term1 = dic1[0]**2/(2*3**2)
    # for ii in np.arange(1,int(args.input_dim/2),1):
    #     term1 = term1 + dic1[ii]**2/(2 * (2.718281828459045**(dic1[0] / 2))**2)
    # term2 = dic1[int(args.input_dim/2)]**2/2
    # for ii in np.arange(int(args.input_dim/2)+1,int(args.input_dim),1):
    #     term2 = term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
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

    return H

L = 40
N = 100
steps = L*600
t_span = [0,L]
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-5}
y0 = np.zeros(args.input_dim)
for ii in np.arange(0,int(args.input_dim/2),1):
    y0[ii] = 1.0
    # if ii == 2:
    #     y0[ii] = 10.0
for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
    y0[ii] = norm(loc=0,scale=1).rvs()
    # if ii == int(args.input_dim/2):
    #     y0[ii] = norm(loc=0,scale=1).rvs()
    # else:
    #     y0[ii] = norm(loc=0,scale=(2.718281828459045**(y0[0] / 2))**(-1)).rvs()
x_req = np.zeros((N,int(args.input_dim/2)))
x_req[0,:] = y0[0:int(args.input_dim/2)]
accept = np.zeros(N)
RK = np.zeros((steps+1,int(args.input_dim),N))
HNN_sto = np.zeros((args.input_dim,steps,N))

for ii in np.arange(0,N-1,1):
    hnn_ivp = integrate_model(hnn_model, t_span, y0, steps, **kwargs)
    # HNN_sto[:,:,ii] = hnn_ivp.y[0:args.input_dim,:]
    HNN_sto[:,:,ii] = hnn_ivp[0:args.input_dim,0:steps]
    yhamil = np.zeros(args.input_dim)
    for jj in np.arange(0,args.input_dim,1):
        # yhamil[jj] = hnn_ivp.y[jj,steps-1]
        yhamil[jj] = hnn_ivp[jj,steps-1]
    H_star = hamil(yhamil)
    H_prev = hamil(y0)
    alpha = np.minimum(1,np.exp(H_prev - H_star))
    if alpha > uniform().rvs():
        # y0[0:int(args.input_dim/2)] = hnn_ivp.y[0:int(args.input_dim/2),steps-1]
        y0[0:int(args.input_dim/2)] = hnn_ivp[0:int(args.input_dim/2),steps-1]
        # x_req[ii+1,:] = hnn_ivp.y[0:int(args.input_dim/2),steps-1]
        x_req[ii+1,:] = hnn_ivp[0:int(args.input_dim/2),steps-1]
        accept[ii+1] = 1
    else:
        x_req[ii+1,:] = y0[0:int(args.input_dim/2)]
    for jj in np.arange(int(args.input_dim/2),args.input_dim,1):
        y0[jj] = norm(loc=0,scale=1).rvs()
        # if ii == int(args.input_dim/2):
        #     y0[jj] = norm(loc=0,scale=1).rvs()
        # else:
        #     y0[jj] = norm(loc=0,scale=(2.718281828459045**(y0[0] / 2))**(-1)).rvs()
    print(ii)

hmc1 = x_req[:,0]
hmc2 = x_req[:,1]
hmc3 = x_req[:,2]
hmc4 = x_req[:,3]
# hmc5 = x_req[:,4]
# plt.scatter(hmc1,hmc2)
# plt.xlabel('x1')
# plt.ylabel('x2')
# x_req1 = x_req[0:2000,:]
df = pd.DataFrame(x_req, columns = ['x1', 'x2', 'x3', 'x4']) # ,'x5', 'x6', 'x7', 'x8', 'x9', 'x10'
scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')

# plt.scatter(hmc1,hmc3)
# plt.xlabel('x1')
# plt.ylabel('x3')

# plt.scatter(hmc1,hmc4)
# plt.xlabel('x1')
# plt.ylabel('x4')

# plt.scatter(hmc1,hmc5)
# plt.xlabel('x1')
# plt.ylabel('x5')

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x_req[0:850,0],x_req[0:850,1],x_req[0:850,2],s=1)
# ax.scatter(0.0,0.0,0.0)
# ax.plot3D(hmc1,hmc2,hmc3,'k',linewidth=0.25)
# ax.scatter(hmc2,hmc3,hmc4,'k',s=1.5)
# ax.scatter(ref2,ref3,ref4,'k',s=1)

rk_req = np.zeros((N,int(args.input_dim/2)))
rk_accept = np.zeros(N)
# for ii in np.arange(0,int(args.input_dim/2),1):
#     y0[ii] = 0.0
# for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
#     y0[ii] = norm(loc=0,scale=1).rvs()
for ii in np.arange(0,N-1,1):
    y0 = HNN_sto[:,0,ii]
    data = get_dataset(y0=y0, samples=1, test_split=1.0)
    RK[:,:,ii] = data.get("coords")[:,0:int(args.input_dim)]
    yhamil1 = np.zeros(args.input_dim)
    for jj in np.arange(0,args.input_dim,1):
        yhamil1[jj] = data.get("coords")[steps-1,jj]
    H_star = hamil(yhamil1)
    H_prev = hamil(y0)
    alpha = np.minimum(1,np.exp(H_prev - H_star))
    if alpha > uniform().rvs():
        # y0[0:int(args.input_dim/2)] = data.get("coords")[steps-1,0:int(args.input_dim/2)]
        rk_req[ii+1,0:int(args.input_dim/2)] = data.get("coords")[steps-1,0:int(args.input_dim/2)]
        rk_accept[ii+1] = 1
    else:
        rk_req[ii+1,:] = y0[0:int(args.input_dim/2)]
    # for jj in np.arange(int(args.input_dim/2),args.input_dim,1):
    #     y0[jj] = norm(loc=0,scale=1).rvs()
    print(ii)

ref1 = rk_req[:,0]
ref2 = rk_req[:,1]
ref3 = rk_req[:,2]
ref4 = rk_req[:,3]
df1 = pd.DataFrame(rk_req[0:1000,:], columns = ['x1', 'x2', 'x3', 'x4'])
scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
scatter_matrix(df1, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # ax.scatter(hmc1,hmc2,hmc3,s=5)
# ax.scatter(ref1,ref2,ref3,s=5)

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(ref1, ref2, label='Numerical gradients')
# ax.scatter(hmc1, hmc2, label='Energy-conserving neural ODEs')
# ax.set_ylabel('y coordinate')
# ax.set_xlabel('x coordinate')
# plt.legend(frameon=False)
# plt.title('2D Rosenbrock density',fontsize=18)
# plt.savefig('Schem2.pdf', format='pdf', bbox_inches = "tight")

idx = 0

fig = plt.figure(figsize=(6, 6))
plt.plot(HNN_sto[0,1:1600,idx],HNN_sto[1,1:1600,idx], label='Hamiltonian neural ODEs')
plt.plot(RK[:,0,idx],RK[:,1,idx], label='Numerical gradients')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(frameon=False)

fig = plt.figure(figsize=(6, 6))
plt.plot(HNN_sto[2,1:1600,idx],HNN_sto[3,1:1600,idx])
plt.plot(RK[:,2,idx],RK[:,3,idx])
plt.xlabel('X3')
plt.ylabel('X4')
plt.legend(frameon=False)

H_HNN = np.zeros(steps)
H_HMC = np.zeros(steps)
for ii in range(0,steps):
    H_HNN[ii] = hamil(HNN_sto[:,ii,idx])
    # H_HMC[ii] = hamil(RK[ii,:,idx])
    
plt.plot(H_HNN)
plt.plot(H_HMC)

# fig = plt.figure(figsize=(6, 6))
# plt.plot(HNN_sto[2,:,0:400], HNN_sto[3,:,0:400],'k',linewidth=0.25)
# plt.scatter(HNN_sto[2,:,0:400], HNN_sto[3,:,0:400],s=0.5, c='k')
# plt.xlabel('X3')
# plt.ylabel('X4')



# fig = plt.figure(figsize=(6, 6))
# plt.plot(HNN_sto[2,1:1600,1],HNN_sto[3,1:1600,1])
# plt.plot(RK[:,2,1],RK[:,3,1])
# plt.xlabel('X3')
# plt.ylabel('X4')
# plt.legend(frameon=False)

