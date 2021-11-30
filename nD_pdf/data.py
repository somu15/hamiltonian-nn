# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np
from scipy.stats import norm
from pyDOE import *
from scipy.stats import uniform

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

input_dim1 = 5
Nsamps = 20
# lhd0 = lhs(1, samples=Nsamps+1, criterion='centermaximin').reshape(Nsamps+1)
# lhd = np.zeros((Nsamps+1,input_dim1))
# lhd[:,0] = norm().ppf(lhd0)
# for ii in  np.arange(1,input_dim1,1):
#     lhd[:,ii] = uniform(loc=-10,scale=20).ppf(lhd0)
# # lhd = uniform(loc=0,scale=25).ppf(lhd0)
# lhd = np.zeros((Nsamps+1,input_dim1))
# lhd[:,0] = norm(loc=0,scale=np.sqrt(1.e-02)).ppf(lhd0)
# lhd[:,1] = norm(loc=0,scale=np.sqrt(1.e-01)).ppf(lhd0)
# lhd[:,2] = norm(loc=0,scale=np.sqrt(1.e+00)).ppf(lhd0)
# lhd[:,3] = norm(loc=0,scale=np.sqrt(1.e+01)).ppf(lhd0)
# lhd[:,4] = norm(loc=0,scale=np.sqrt(1.e+02)).ppf(lhd0)

def hamiltonian_fn(coords):

    #******** 5D Gaussian #********
    # dic1 = np.split(coords,2*input_dim1)
    # var1 = np.array([1.,1.,1.,1.,1.])
    # term1 = dic1[0]**2/(2*var1[0])
    # for ii in np.arange(1,5,1):
    #     term1 = term1 + dic1[ii]**2/(2*var1[ii])
    # term2 = dic1[5]**2/2
    # for ii in np.arange(6,10,1):
    #     term2 = term2 + dic1[ii]**2/2
    # H = term1 + term2

    #******** 2D Example from Rashki MSSP #********
    # dic1 = np.split(coords,2*input_dim1)
    # # tmp1 = ((30 / (4 * (dic1[0] + 2)**2) / 9) + (dic1[1]**2 / 25)**2 + 1)
    # # tmp2 = (20 / (((dic1[0] - 2.5)**2 / 4) + ((dic1[1] - 0.5)**2 / 25)**2 + 1) - 5)
    # # term1 = tmp1 + tmp2
    # tmp1 = (4 - dic1[0]) * (dic1[0] > 3.5) + (0.85 - 0.1 * dic1[0]) * (dic1[0] <= 3.5)
    # tmp2 = (4 - dic1[1]) * (dic1[1] > 3.5) + (0.85 - 0.1 * dic1[1]) * (dic1[1] <= 3.5)
    # term1 = tmp1 * (tmp1 < tmp2) + tmp2 * (tmp2 < tmp1)
    # term2 = dic1[2]**2/2 + dic1[3]**2/2
    # H = term1 + term2

    #******** 5D Ill-Conditioned Gaussian #********
    dic1 = np.split(coords,2*input_dim1)
    var1 = np.array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])
    term1 = dic1[0]**2/(2*var1[0])
    for ii in np.arange(1,5,1):
        term1 = term1 + dic1[ii]**2/(2*var1[ii])
    term2 = dic1[5]**2/2
    for ii in np.arange(6,10,1):
        term2 = term2 + dic1[ii]**2/2
    H = term1 + term2

    #******** 2D Funnel #********
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = dic1[0]**2/(2*3**2)
    # for ii in np.arange(1,2,1):
    #     term1 = term1 + dic1[ii]**2/(2 * (2.718281828459045**(dic1[0] / 2))**2)
    # term2 = dic1[2]**2/2
    # for ii in np.arange(3,4,1):
    #     term2 = term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
    # H = term1 + term2

    #******** 10D Funnel #********
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = dic1[0]**2/(2*3**2)
    # for ii in np.arange(1,10,1):
    #     term1 = term1 + dic1[ii]**2/(2 * 2.718281828459045**(-2 * dic1[0]))
    # term2 = dic1[10]**2/2
    # for ii in np.arange(11,20,1):
    #     term2 = term2 + dic1[ii]**2/2
    # H = term1 + term2

    #******** 2D Rosenbrock #********
    # dic1 = np.split(coords,2*input_dim1)
    # a = 1
    # b = 100
    # p = 20
    # term1 = (b*(dic1[1]-dic1[0]**2)**2+(a-dic1[0])**2)/p
    # term2 = 1*dic1[2]**2/2+1*dic1[3]**2/2
    # H = term1 + term2

    #******** 3D Rosenbrock #********
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = (100 * (dic1[1] - dic1[0]**2)**2 + (1 - dic1[0])**2 + 100 * (dic1[2] - dic1[1]**2)**2 + (1 - dic1[1]**2)) / 20
    # term2 = 1*dic1[3]**2/2+1*dic1[4]**2/2+1*dic1[5]**2/2
    # H = term1 + term2

    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dic1 = np.split(dcoords,2*input_dim1)
    S = np.concatenate([dic1[input_dim1]])
    for ii in np.arange(input_dim1+1,2*input_dim1,1):
        S = np.concatenate([S, dic1[ii]])
    for ii in np.arange(0,input_dim1,1):
        S = np.concatenate([S, -dic1[ii]])
    return S

def get_trajectory(t_span=[0,4], timescale=400, radius=None, y0=None, noise_std=0.01, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    if y0 is None:
        y0 = np.zeros(input_dim1)
        for ii in np.arange(0,input_dim1,1):
            y0[ii] = norm(loc=0,scale=1).rvs()
    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs) #  method='RK45',
    # q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = spring_ivp['y'][0], spring_ivp['y'][1], spring_ivp['y'][2], spring_ivp['y'][3], spring_ivp['y'][4], spring_ivp['y'][5], spring_ivp['y'][6], spring_ivp['y'][7], spring_ivp['y'][8], spring_ivp['y'][9], spring_ivp['y'][10], spring_ivp['y'][11], spring_ivp['y'][12], spring_ivp['y'][13], spring_ivp['y'][14], spring_ivp['y'][15], spring_ivp['y'][16], spring_ivp['y'][17], spring_ivp['y'][18], spring_ivp['y'][19]
    dic1 = np.split(spring_ivp['y'], 2*input_dim1)
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    # dq1dt, dq2dt, dq3dt, dq4dt, dq5dt, dq6dt, dq7dt, dq8dt, dq9dt, dq10dt, dp1dt, dp2dt, dp3dt, dp4dt, dp5dt, dp6dt, dp7dt, dp8dt, dp9dt, dp10dt = np.split(dydt,20)
    ddic1 = np.split(dydt, 2*input_dim1)

    # add noise
    # q += np.random.randn(*q.shape)*noise_std
    # p += np.random.randn(*p.shape)*noise_std
    return dic1, ddic1, t_eval

def get_dataset(seed=0, samples=Nsamps, test_split=1.0, **kwargs):
    data = {'meta': locals()}
    # randomly sample inputs
    np.random.seed(seed) #
    xs, dxs = [], []
    index1 = 0

    count1 = 0
    y_init = np.zeros(2*input_dim1)
    for ii in np.arange(0,input_dim1,1):
        y_init[ii] = 0.0 # lhd[count1,ii] # 0.0 #
    for ii in np.arange(input_dim1,2*input_dim1,1):
        # y_init[ii] = norm(loc=0,scale=1).rvs()
        if ii == input_dim1:
            y_init[ii] = norm(loc=0,scale=1).rvs()
        else:
            y_init[ii] = norm(loc=0,scale=(2.718281828459045**(y_init[0] / 2))**(-1)).rvs()

    for s in range(samples):
        dic1, ddic1, t = get_trajectory(y0=y_init,**kwargs) #
        xs.append(np.stack( [dic1[ii].T.reshape(len(dic1[ii].T)) for ii in np.arange(0,2*input_dim1,1)]).T)
        dxs.append(np.stack( [ddic1[ii].T.reshape(len(ddic1[ii].T)) for ii in np.arange(0,2*input_dim1,1)]).T)
        y_init = np.zeros(2*input_dim1)
        count1 = count1 + 1
        for ii in np.arange(0,input_dim1,1):
            y_init[ii] = dic1[ii].T[len(dic1[ii].T)-1] # lhd[count1,ii] #
        for ii in np.arange(input_dim1,2*input_dim1,1):
            # y_init[ii] = norm(loc=0,scale=1).rvs()
            if ii == input_dim1:
                y_init[ii] = norm(loc=0,scale=1).rvs()
            else:
                y_init[ii] = norm(loc=0,scale=(2.718281828459045**(y_init[0] / 2))**(-1)).rvs()

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
