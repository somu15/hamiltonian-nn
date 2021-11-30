#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 2021

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

EXPERIMENT_DIR = './nD_pdf'
sys.path.append(EXPERIMENT_DIR)

from data import get_dataset, get_field, get_trajectory, dynamics_fn, hamiltonian_fn
from nn_models import MLP
from hnn import HNN
from utils import L2_loss
from scipy.stats import norm
from scipy.stats import uniform

DPI = 300
FORMAT = 'pdf'
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2
RK4 = ''

## Ref: 50; 2 and 200

def get_args():
    return {'input_dim': 10,
         'hidden_dim': 50,
         'learn_rate': 5e-4,
         'nonlinearity': 'tanh',
         'total_steps': 7000,
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

def get_vector_field(model, **kwargs):
    field = get_field(**kwargs)
    np_mesh_x = field['x']

    # run model
    mesh_x = torch.tensor( np_mesh_x, requires_grad=True, dtype=torch.float32)
    mesh_dx = model.time_derivative(mesh_x)
    return mesh_dx.data.numpy()

def integrate_model(model, t_span, y0, **kwargs):

    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,args.input_dim)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

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
    dic1 = np.split(coords,args.input_dim)
    var1 = np.array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])
    term1 = dic1[0]**2/(2*var1[0])
    for ii in np.arange(1,5,1):
        term1 = term1 + dic1[ii]**2/(2*var1[ii])
    term2 = dic1[5]**2/2
    for ii in np.arange(6,10,1):
        term2 = term2 + dic1[ii]**2/2
    H = term1 + term2
    
    #******** 2D Funnel #********
    # dic1 = np.split(coords,args.input_dim)
    # term1 = dic1[0]**2/(2*3**2)
    # for ii in np.arange(1,2,1):
    #     term1 = term1 + dic1[ii]**2/(2 * (2.718281828459045**(dic1[0] / 2))**2)
    # term2 = dic1[2]**2/2
    # for ii in np.arange(3,4,1):
    #     term2 = term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
    # H = term1 + term2
    
    #******** 10D Funnel #********
    # dic1 = np.split(coords,args.input_dim)
    # term1 = dic1[0]**2/(2*3**2)
    # for ii in np.arange(1,10,1):
    #     term1 = term1 + dic1[ii]**2/(2 * 2.718281828459045**(-2 * dic1[0]))
    # term2 = dic1[10]**2/2
    # for ii in np.arange(11,20,1):
    #     term2 = term2 + dic1[ii]**2/2
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

    return H

L = 4
N = 2000
steps = L*400
t_span = [0,L]
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-5}
y0 = np.zeros(args.input_dim)
for ii in np.arange(0,int(args.input_dim/2),1):
    y0[ii] = 0.0
    # if ii == 2:
    #     y0[ii] = 10.0
for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
    y0[ii] = norm(loc=0,scale=1).rvs()
x_req = np.zeros((N,int(args.input_dim/2)))
x_req[0,:] = y0[0:int(args.input_dim/2)]
accept = np.zeros(N)
RK = np.zeros((steps,int(args.input_dim/2),N))
HNN_sto = np.zeros((args.input_dim,steps,N))

for ii in np.arange(0,N-1,1):
    hnn_ivp = integrate_model(hnn_model, t_span, y0, **kwargs)
    HNN_sto[:,:,ii] = hnn_ivp.y[0:args.input_dim,:]
    yhamil = np.zeros(args.input_dim)
    for jj in np.arange(0,args.input_dim,1):
        yhamil[jj] = hnn_ivp.y[jj,steps-1]
    H_star = hamil(yhamil)
    H_prev = hamil(y0)
    alpha = np.minimum(1,np.exp(H_prev - H_star))
    if alpha > uniform().rvs():
        y0[0:int(args.input_dim/2)] = hnn_ivp.y[0:int(args.input_dim/2),steps-1]
        x_req[ii+1,:] = hnn_ivp.y[0:int(args.input_dim/2),steps-1]
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

# hmc1 = x_req[:,0]
# hmc2 = x_req[:,1]
# hmc3 = x_req[:,2]
# hmc4 = x_req[:,3]
# hmc5 = x_req[:,4]
# plt.scatter(hmc1,hmc2)
# plt.xlabel('x1')
# plt.ylabel('x2')

# plt.scatter(hmc1,hmc3)
# plt.xlabel('x1')
# plt.ylabel('x3')

# plt.scatter(hmc1,hmc4)
# plt.xlabel('x1')
# plt.ylabel('x4')

# plt.scatter(hmc1,hmc5)
# plt.xlabel('x1')
# plt.ylabel('x5')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.scatter(hmc1,hmc2,hmc3,s=1)
ax.scatter(0.0,0.0,0.0)
ax.plot3D(hmc1,hmc2,hmc3,'k',linewidth=0.25)

rk_req = np.zeros((N,int(args.input_dim/2)))
rk_accept = np.zeros(N)
# for ii in np.arange(0,int(args.input_dim/2),1):
#     y0[ii] = 0.0
# for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
#     y0[ii] = norm(loc=0,scale=1).rvs()
for ii in np.arange(0,N-1,1):
    y0 = HNN_sto[:,0,ii]
    data = get_dataset(y0=y0, samples=1, test_split=1.0)
    RK[:,:,ii] = data.get("coords")[:,0:int(args.input_dim/2)]
    yhamil1 = np.zeros(args.input_dim)
    for jj in np.arange(0,args.input_dim,1):
        yhamil1[jj] = data.get("coords")[steps-1,jj]
    H_star = hamil(yhamil1)
    H_prev = hamil(y0)
    alpha = np.minimum(1,np.exp(H_prev - H_star))
    if alpha > uniform().rvs():
        y0[0:int(args.input_dim/2)] = data.get("coords")[steps-1,0:int(args.input_dim/2)]
        rk_req[ii+1,0:int(args.input_dim/2)] = data.get("coords")[steps-1,0:int(args.input_dim/2)]
        rk_accept[ii+1] = 1
    else:
        rk_req[ii+1,:] = y0[0:int(args.input_dim/2)]
    for jj in np.arange(int(args.input_dim/2),args.input_dim,1):
        y0[jj] = norm(loc=0,scale=1).rvs()
    print(ii)

ref1 = rk_req[:,0]
ref2 = rk_req[:,1]
# ref3 = rk_req[:,2]

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # ax.scatter(hmc1,hmc2,hmc3,s=5)
# ax.scatter(ref1,ref2,ref3,s=5)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(ref1, ref2, label='Numerical gradients')
ax.scatter(hmc1, hmc2, label='Energy-conserving neural ODEs')
ax.set_ylabel('y coordinate')
ax.set_xlabel('x coordinate')
plt.legend(frameon=False)
plt.title('2D Rosenbrock density',fontsize=18)
plt.savefig('Schem2.pdf', format='pdf', bbox_inches = "tight")



