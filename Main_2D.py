#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:23:46 2021

@author: dhulls
"""

import torch, time, sys
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

EXPERIMENT_DIR = './2D_pdf'
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

def get_args():
    return {'input_dim': 4,
         'hidden_dim': 20,
         'learn_rate': 1e-3,
         'nonlinearity': 'tanh',
         'total_steps': 10000,
         'field_type': 'solenoidal',
         'print_every': 200,
         'name': '2dpdf',
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
    output_dim = args.input_dim if baseline else 4
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=baseline)

    model_name = 'baseline' if baseline else 'hnn'
    path = "{}/2dpdf{}-{}.tar".format(args.save_dir, RK4, model_name)
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
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,4)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

# base_model = get_model(args, baseline=False)
hnn_model = get_model(args, baseline=False)

# get their vector fields
# R = 2.6
# field = get_field(xmin=-R, xmax=R, ymin=-R, ymax=R, gridsize=args.gridsize)
# data = get_dataset(radius=2.0)
# base_field = get_vector_field(base_model, xmin=-R, xmax=R, ymin=-R, ymax=R, gridsize=args.gridsize)
# hnn_field = get_vector_field(hnn_model, xmin=-R, xmax=R, ymin=-R, ymax=R, gridsize=args.gridsize)

# integrate along those fields starting from point (1,0)
# t_span = [0,20]
# y0 = np.asarray([2.1, 0])
# kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 1000), 'rtol': 1e-12}
# # base_ivp = integrate_model(base_model, t_span, y0, **kwargs)
# hnn_ivp = integrate_model(hnn_model, t_span, y0, **kwargs)

## HMC with HNN

def hamil(coords):
    q1, q2, p1, p2 = np.split(coords,4)
    a = 1
    b = 100
    p = 20
    term1 = (b*(q2-q1**2)**2+(a-q1)**2)/p
    term2 = p1**2/2+p2**2/2
    H = term1 + term2
    return H

L = 4
N = 100
steps = L*400
t_span = [0,L]
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-3}
y0 = np.array([2.0, 2.0, norm(loc=0,scale=1).rvs(), norm(loc=0,scale=1).rvs()]) # uniform().rvs()*3.-3.
x_req = np.zeros((N,2))
x_req[0,:] = y0[0:2]
accept = np.zeros(N)
RK = np.zeros((steps,2,N))
HNN_sto = np.zeros((4,steps,N))

for ii in np.arange(0,N-1,1):
    hnn_ivp = integrate_model(hnn_model, t_span, y0, **kwargs)
    # data = get_dataset(y0=y0, samples=1, test_split=1.0)
    # RK[:,:,ii] = data.get("coords")[:,0:2]
    HNN_sto[:,:,ii] = hnn_ivp.y[0:4,:]
    # H_star = hnn_ivp.y[0,steps-1]**2/2 + hnn_ivp.y[1,steps-1]**2/2
    # H_prev = y0[0]**2/2 + y0[1]**2/2
    H_star = hamil(np.array([hnn_ivp.y[0,steps-1], hnn_ivp.y[1,steps-1],hnn_ivp.y[2,steps-1], hnn_ivp.y[3,steps-1]]))
    H_prev = hamil(y0)
    alpha = np.minimum(1,np.exp(H_prev - H_star))
    if alpha > uniform().rvs():
        y0[0:2] = hnn_ivp.y[0:2,steps-1]
        x_req[ii+1,:] = hnn_ivp.y[0:2,steps-1]
        accept[ii+1] = 1
    else:
        x_req[ii+1,:] = y0[0:2]
    y0[2] = norm(loc=0,scale=1).rvs() # uniform().rvs()*3.-3. #
    y0[3] = norm(loc=0,scale=1).rvs()
    print(ii)

rk_req = np.zeros((N,2))
rk_accept = np.zeros(N)
for ii in np.arange(0,N-1,1):
    y0 = HNN_sto[:,0,ii]
    data = get_dataset(y0=y0, samples=1, test_split=1.0)
    RK[:,:,ii] = data.get("coords")[:,0:2]
    H_star = hamil(np.array([data.get("coords")[steps-1,0], data.get("coords")[steps-1,1],data.get("coords")[steps-1,2], data.get("coords")[steps-1,3]]))
    H_prev = hamil(y0)
    alpha = np.minimum(1,np.exp(H_prev - H_star))
    if alpha > uniform().rvs():
        rk_req[ii+1,0] = data.get("coords")[steps-1,0]
        rk_req[ii+1,1] = data.get("coords")[steps-1,1]
        rk_accept[ii+1] = 1
    else:
        rk_req[ii+1,:] = y0[0:2]
    print(ii)

plt.scatter(rk_req[:,0],rk_req[:,1],label='RK 4')
plt.scatter(x_req[:,0],x_req[:,1], label='Hamil. NNs')
plt.xlabel('x-coord')
plt.ylabel('y-coord')
plt.legend()

# ind_req = 50
# plt.plot(RK[:,0,ind_req],RK[:,1,ind_req],label='RK 4')
# plt.plot(HNN_sto[0,:,ind_req],HNN_sto[1,:,ind_req], label='Hamil. NNs')
# plt.xlabel('x-coord')
# plt.ylabel('y-coord')
# plt.legend()

# plt.plot(data.get("coords")[:,0],data.get("coords")[:,1], label='RK4')
# plt.plot(hnn_ivp.y[0,:],hnn_ivp.y[1,:], label='Hamil. NNs')

###### PLOT ######
# fig = plt.figure(figsize=(11.3, 3.2), facecolor='white', dpi=DPI)

# # plot physical system
# fig.add_subplot(1, 4, 1, frameon=True)
# plt.xticks([]) ;  plt.yticks([])
# schema = mpimg.imread(EXPERIMENT_DIR + '/pendulum.png')
# plt.imshow(schema)
# plt.title("Pendulum system", pad=10)

# # plot dynamics
# fig.add_subplot(1, 4, 2, frameon=True)
# x, y, dx, dy, t = get_trajectory(t_span=[0,28], radius=2.1, noise_std=0.01, y0=y0)
# N = len(x)
# point_colors = [(i/N, 0, 1-i/N) for i in range(N)]
# plt.scatter(x,y, s=14, label='data', c=point_colors)

# plt.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
#         cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2))
# plt.xlabel("$q$", fontsize=14)
# plt.ylabel("$p$", rotation=0, fontsize=14)
# plt.title("Data", pad=10)

# # plot baseline
# fig.add_subplot(1, 4, 4, frameon=True)
# plt.quiver(field['x'][:,0], field['x'][:,1], base_field[:,0], base_field[:,1],
#         cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.5,.5,.5))

# for i, l in enumerate(np.split(base_ivp['y'].T, LINE_SEGMENTS)):
#     color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
#     plt.plot(l[:,0],l[:,1],color=color, linewidth=LINE_WIDTH)

# plt.xlabel("$q$", fontsize=14)
# plt.ylabel("$p$", rotation=0, fontsize=14)
# plt.title("Baseline NN", pad=10)

# # plot HNN
# fig.add_subplot(1, 4, 4, frameon=True)
# plt.quiver(field['x'][:,0], field['x'][:,1], hnn_field[:,0], hnn_field[:,1],
#         cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.5,.5,.5))

# for i, l in enumerate(np.split(hnn_ivp['y'].T, LINE_SEGMENTS)):
#     color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
#     plt.plot(l[:,0],l[:,1],color=color, linewidth=LINE_WIDTH)

# plt.xlabel("$q$", fontsize=14)
# plt.ylabel("$p$", rotation=0, fontsize=14)
# plt.title("Hamiltonian NN", pad=10)

# plt.tight_layout() ; plt.show()
# fig.savefig('{}/pend{}.{}'.format(args.fig_dir, RK4, FORMAT))
