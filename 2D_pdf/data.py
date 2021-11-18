# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np
from scipy.stats import norm
from pyDOE import *
from scipy.stats import uniform

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

Nsamps = 10
# lhd0 = lhs(2, samples=Nsamps+1, criterion='centermaximin')
# lhd = uniform(loc=0,scale=25).ppf(lhd0)
# lhd = np.zeros((Nsamps+1,2))
# lhd[:,0] = uniform(loc=-3,scale=6).rvs(Nsamps+1)
# lhd[:,1] = uniform(loc=0,scale=25).rvs(Nsamps+1)

# def taylor_sine(x):  # Taylor approximation to sine function
#     ans = currterm = x
#     i = 0
#     while np.abs(currterm) > 0.001:
#         currterm = -currterm * x**2 / ((2 * i + 3) * (2 * i + 2))
#         ans = ans + currterm
#         i += 1
#     return ans
#
# def taylor_cosine(x):  # Taylor approximation to cosine function
#     ans = currterm = 1
#     i = 0
#     while np.abs(currterm) > 0.001:
#         currterm = -currterm * x**2 / ((2 * i + 1) * (2 * i + 2))
#         ans = ans + currterm
#         i += 1
#     return ans

def hamiltonian_fn(coords):

    #******** Multimodal Himmelblau function #********
    # q1, q2, p1, p2 = np.split(coords,4)
    # term1 = 0.1 * ((q1**2 + q2 - 11)**2 + (q1 + q2**2 - 7)**2)
    # term2 = 1*p1**2/2+1*p2**2/2
    # H = term1 + term2

    #******** 2D Gausssian #********
    # q1, q2, p1, p2 = np.split(coords,4)
    # mu = np.array([0.,0.])
    # sigma_inv = np.linalg.inv(np.array([[100.0,2.483],[2.483,0.1]]))
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = (y[0]*tmp1[0] + y[1]*tmp1[1])/2
    # term2 = p1**2/2+p2**2/2
    # H = term1 + term2

    #******** 2D Gausssian (4 mixtures) #********
    # q1, q2, p1, p2 = np.split(coords,4)
    # sigma_inv = np.array([[1.,0.],[0.,1.]])
    # term1 = 0.
    #
    # mu = np.array([3.,0.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    #
    # mu = np.array([-3.,0.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    #
    # mu = np.array([0.,3.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    #
    # mu = np.array([0.,-3.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    #
    # term1 = -np.log(term1)
    # term2 = p1**2/2+p2**2/2
    # H = term1 + term2

    #******** 2D Rosenbrock #********
    q1, q2, p1, p2 = np.split(coords,4)
    a = 1
    b = 100
    p = 20
    term1 = (b*(q2-q1**2)**2+(a-q1)**2)/p
    term2 = 1*p1**2/2+1*p2**2/2
    H = term1 + term2

    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dq1dt, dq2dt, dp1dt, dp2dt = np.split(dcoords,4)
    S = np.concatenate([dp1dt, dp2dt, -dq1dt, -dq2dt], axis=-1)
    return S

def get_trajectory(t_span=[0,4], timescale=400, radius=None, y0=None, noise_std=0.01, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    if y0 is None:
        y0 = np.array([0.,0.,0.,0.])
        y0[0] = norm(loc=0,scale=1).rvs()
        y0[1] = norm(loc=0,scale=1).rvs()
        y0[2] = norm(loc=0,scale=1).rvs()
        y0[3] = norm(loc=0,scale=1).rvs()

    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q1, q2, p1, p2 = spring_ivp['y'][0], spring_ivp['y'][1], spring_ivp['y'][2], spring_ivp['y'][3]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dq1dt, dq2dt, dp1dt, dp2dt = np.split(dydt,4)

    # add noise
    # q += np.random.randn(*q.shape)*noise_std
    # p += np.random.randn(*p.shape)*noise_std
    return q1, q2, p1, p2, dq1dt, dq2dt, dp1dt, dp2dt, t_eval

def get_dataset(seed=0, samples=Nsamps, test_split=1.0, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed) #
    xs, dxs = [], []
    index1 = 0
    # y_init = np.array([lhd[index1,0],lhd[index1,1], norm(loc=0,scale=1.).rvs(), norm(loc=0,scale=1.).rvs()])
    y_init = np.array([1.,1., norm(loc=0,scale=1.).rvs(), norm(loc=0,scale=1.).rvs()])
    for s in range(samples):
        x1, x2, y1, y2, dx1, dx2, dy1, dy2, t = get_trajectory(y0=y_init, **kwargs) #
        xs.append( np.stack( [x1, x2, y1, y2]).T )
        dxs.append( np.stack( [dx1, dx2, dy1, dy2]).T ) # hnn_ivp.y[0:2,steps-1]
        y_init = np.array([x1[len(x1)-1], x2[len(x2)-1], norm(loc=0,scale=1).rvs(), norm(loc=0,scale=1).rvs()]) # +norm(loc=0.01,scale=1).rvs() +norm(loc=0.01,scale=1).rvs()
        # index1 = index1 + 1
        # y_init = np.array([lhd[index1,0],lhd[index1,1], norm(loc=0,scale=1.).rvs(), norm(loc=0,scale=1.).rvs()])

    data['coords'] = np.concatenate(xs)
    data['dcoords'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['coords']) * test_split)
    split_data = {}
    for k in ['coords', 'dcoords']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])

    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field
