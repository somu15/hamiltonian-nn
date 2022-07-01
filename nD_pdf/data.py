# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski
import csv
import autograd.numpy as np
# import jax.numpy as np
import autograd
# from jax import grad
from scipy.stats import norm
from pyDOE import *
from scipy.stats import uniform
from scipy.stats import wishart

# from Test_Data import func1, NODE_hamil, getgrad_NODE

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from utils import to_pickle, from_pickle
import pickle

input_dim1 = 100
Nsamps = 200
# lhd = np.zeros((Nsamps+1,input_dim1))
# lhd[0,:] = np.array([-4.81365419e+00, -2.97798186e-02])
# lhd[1,:] = np.array([-5.62375103e+00,  5.59615650e-02])
# lhd[2,:] = np.array([-3.13534412e+00, -7.82559226e-03])
# lhd[3,:] = np.array([-4.10408610e+00, -1.75231961e-01])
# lhd[4,:] = np.array([-3.48652441e+00,  2.08413696e-02])
# lhd[5,:] = np.array([-1.89749715, -0.17589146])
# lhd[6,:] = np.array([-0.99256864, -0.00727149])
# lhd[7,:] = np.array([-2.9187646 ,  0.05156044])
# lhd[8,:] = np.array([-0.30902009, -0.14061672])
# lhd[9,:] = np.array([-0.74445954,  1.93901093])
# lhd[10,:] = np.array([-2.94024525, -0.13851359])
# lhd[11,:] = np.array([-2.89546373,  0.19974407])
# lhd[12,:] = np.array([-0.46497868,  0.10666009])
# lhd[13,:] = np.array([-0.3279123 , -0.60457858])
# lhd[14,:] = np.array([-1.48711408, -0.22847326])
# lhd[15,:] = np.array([ 2.66950842, -1.87534805])
# lhd[16,:] = np.array([ 3.88019759,  3.28248355])
# lhd[17,:] = np.array([ 2.53041437,  5.54543447])
# lhd[18,:] = np.array([ 4.05471661, -8.08981748])
# lhd[19,:] = np.array([ 3.4939906 , -0.402581  ])
# lhd[20,:] = np.array([ 4.63787144, 11.9996081 ])
# lhd[21,:] = np.array([ 4.52431852, 11.81929023])
# lhd[22,:] = np.array([ 3.15674207,  1.03361234])
# lhd[24,:] = np.array([ 4.79236031, 15.95186538])
# lhd[25,:] = np.array([  6.73753264, -37.0528468 ])
# lhd[26,:] = np.array([  5.81396227, -12.01195105])
# lhd[27,:] = np.array([  6.77140021, -16.44087366])
# lhd[28,:] = np.array([  6.49234703, -33.48267789])
# lhd[29,:] = np.array([  5.37633032,  18.07759346])
# lhd[30,:] = np.array([  5.90846514, -15.15809784])
# lhd[31,:] = np.array([  8.83140151,   0.62088719])
# lhd[32,:] = np.array([  7.30044902,  82.16536546])
# lhd[33,:] = np.array([  6.85262912,  35.97655317])
# lhd[34,:] = np.array([  8.24849661, 137.27889409])
# lhd[23,:] = np.array([ 11.10258934, 505.95724823])
# lhd[35,:] = np.array([ 10.64813531, 233.35850515])
# lhd[36,:] = np.array([ 10.86277996, 346.52612434])
# lhd[37,:] = np.array([  10.17857642, -394.79699817])
# lhd[38,:] = np.array([  10.33035825, -203.99648423])
# lhd[39,:] = np.array([  10.52160846, -312.74337054])
# lhd[40,:] = np.array([ 1.,1.])

# lhd0 = lhs(1, samples=Nsamps+1, criterion='centermaximin').reshape(Nsamps+1)
# lhd = np.zeros((Nsamps+1,input_dim1))
# lhd[:,0] = uniform(loc=-3,scale=6).rvs(Nsamps+1)
# lhd[:,1] = uniform(loc=0,scale=6).rvs(Nsamps+1)
# lhd[:,2] = uniform(loc=0,scale=60).rvs(Nsamps+1)
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

    #******** 1D Gaussian Mixture #********
    # q, p = np.split(coords,2)
    # mu1 = 1.0
    # mu2 = -1.0
    # sigma = 0.35
    # term1 = -np.log(0.5*(np.exp(-(q-mu1)**2/(2*sigma**2)))+0.5*(np.exp(-(q-mu2)**2/(2*sigma**2))))
    # H = term1 + p**2/2 # Normal PDF

    # #******** 2D Gaussian Four Mixtures #********
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
    #
    # mu = np.array([0.,0.])
    # y = np.array([q1-mu[0],q2-mu[1]])
    # tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    # term1 = term1 + np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    #
    # term1 = -np.log(term1)
    # term2 = p1**2/2+p2**2/2
    # H = term1 + term2

    # ******** 5D Gaussian #********
    # dic1 = np.split(coords,2*input_dim1)
    # var1 = np.array([1.,1.,1.,1.,1.])
    # term1 = dic1[0]**2/(2*var1[0])
    # for ii in np.arange(1,5,1):
    #     term1 = term1 + dic1[ii]**2/(2*var1[ii])
    # term2 = dic1[5]**2/2
    # for ii in np.arange(6,10,1):
    #     term2 = term2 + dic1[ii]**2/2
    # H = term1 + term2
    #
    # ******** 2D Example from Rashki MSSP #********
    # dic1 = np.split(coords,2*input_dim1)
    # # tmp1 = ((30 / (4 * (dic1[0] + 2)**2) / 9) + (dic1[1]**2 / 25)**2 + 1)
    # # tmp2 = (20 / (((dic1[0] - 2.5)**2 / 4) + ((dic1[1] - 0.5)**2 / 25)**2 + 1) - 5)
    # # term1 = tmp1 + tmp2
    # tmp1 = (4 - dic1[0]) * (dic1[0] > 3.5) + (0.85 - 0.1 * dic1[0]) * (dic1[0] <= 3.5)
    # tmp2 = (4 - dic1[1]) * (dic1[1] > 3.5) + (0.85 - 0.1 * dic1[1]) * (dic1[1] <= 3.5)
    # term1 = tmp1 * (tmp1 < tmp2) + tmp2 * (tmp2 < tmp1)
    # term2 = dic1[2]**2/2 + dic1[3]**2/2
    # H = term1 + term2
    #
    # ******** 5D Ill-Conditioned Gaussian #********
    # dic1 = np.split(coords,2*input_dim1)
    # var1 = np.array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])
    # term1 = dic1[0]**2/(2*var1[0])
    # for ii in np.arange(1,5,1):
    #     term1 = term1 + dic1[ii]**2/(2*var1[ii])
    # term2 = dic1[5]**2/2
    # for ii in np.arange(6,10,1):
    #     term2 = term2 + dic1[ii]**2/2
    # H = term1 + term2

    # ******** 100D Gaussian by Radford Neal #********
    # dic1 = np.split(coords,2*input_dim1)
    # var1 =  np.ones(input_dim1) # np.arange(0.01,1.01,0.01)
    # term1 = 0.0
    # for ii in np.arange(0,input_dim1,1):
    #     term1 = term1 + dic1[ii]**2/(2*var1[ii]**2)
    # term2 = 0.0
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + dic1[ii]**2/2
    # H = term1 + term2

    #******** 100D Allen-Cahn #********
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = 0.0
    # h = 1/(input_dim1)
    # for ii in np.arange(0,input_dim1-1,1):
    #     tmp1 = (1-dic1[ii+1]**2)**2
    #     tmp2 =  (1-dic1[ii]**2)**2
    #     term1 = term1 + 1/(2*h) * (dic1[ii+1] - dic1[ii])**2 + h/2 * (tmp1 + tmp2)
    #     # tmp1 = dic1[ii+1] + dic1[ii]
    #     # term1 = term1 + 1/(2*h) * (dic1[ii+1] - dic1[ii])**2 + h/2 * (1 - tmp1**2)**2
    # term2 = 0.0
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + 1*dic1[ii]**2/2
    # H = term1 + term2

    #
    # ******** 2D Funnel #********
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = dic1[0]**2/(2*3**2)
    # for ii in np.arange(1,2,1):
    #     term1 = term1 + dic1[ii]**2/(2 * (2.718281828459045**(dic1[0] / 2))**2)
    # term2 = 0.0
    # for ii in np.arange(2,4,1):
    #     term2 = term2 + dic1[ii]**2/2 # term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
    # H = term1 + term2
    #
    # ******** nD Funnel #********
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = dic1[0]**2/(2*3**2)
    # for ii in np.arange(1,input_dim1,1):
    #     term1 = term1 + dic1[ii]**2/(2 * (2.718281828459045**(dic1[0] / 2))**2)
    # term2 = 0.0
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + dic1[ii]**2/2 # term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
    # H = term1 + term2

    # ********* nD Heirarchical (https://crackedbassoon.com/writing/funneling) *********
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = dic1[0]**2/2
    # term1 = term1 - np.log(2 / (np.pi * (1 + np.exp(dic1[1])**2)))
    # for ii in np.arange(2,input_dim1,1):
    #     term1 = term1 + (dic1[ii] - dic1[0])**2 / (2 * np.exp(dic1[1])**2)
    # term2 = 0.0
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + dic1[ii]**2/2 # term2 + (dic1[ii]**2 * (2.718281828459045**(dic1[0] / 2))**2)/2
    # H = term1 + term2
    #
    # ******** 2D Rosenbrock #********
    # dic1 = np.split(coords,2*input_dim1)
    # a = 1
    # b = 100
    # p = 20
    # term1 = (b*(dic1[1]-dic1[0]**2)**2+(a-dic1[0])**2)/p
    # term2 = 1*dic1[2]**2/2+1*dic1[3]**2/2
    # H = term1 + term2
    #
    # ******** 3D Rosenbrock #********
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = (100 * (dic1[1] - dic1[0]**2)**2 + (1 - dic1[0])**2 + 100 * (dic1[2] - dic1[1]**2)**2 + (1 - dic1[1]**2)) / 20
    # term2 = 1*dic1[3]**2/2+1*dic1[4]**2/2+1*dic1[5]**2/2
    # H = term1 + term2
    #
    # ******** nD Rosenbrock #********
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = 0.0
    # for ii in np.arange(0,input_dim1-1,1):
    #     term1 = term1 + (100.0 * (dic1[ii+1] - dic1[ii]**2)**2 + (1 - dic1[ii])**2) / 20.0
    # term2 = 0.0
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + 1*dic1[ii]**2/2
    # H = term1 + term2

    # ******** nD standard Gaussian #********
    # dic1 = np.split(coords,2*input_dim1)
    # var1 = np.ones(int(input_dim1))
    # term1 = dic1[0]**2/(2*var1[0])
    # for ii in np.arange(1,int(input_dim1),1):
    #     term1 = term1 + dic1[ii]**2/(2*var1[ii])
    # term2 = dic1[int(input_dim1)]**2/2
    # for ii in np.arange(int(input_dim1)+1,int(2*input_dim1),1):
    #     term2 = term2 + dic1[ii]**2/2
    # H = term1 + term2

    # ******** nD Gaussian with Wishart covariance #********
    # dic1 = np.split(coords,2*input_dim1)
    # with open('/Users/dhulls/Desktop/outfile.txt', 'r') as f:
    #     sigma_inv = [[float(num) for num in line.split(',')] for line in f]
    # sigma_inv = np.matrix(sigma_inv)
    # term1 = np.array(0.5 * np.array(dic1[0:int(input_dim1)]).T * sigma_inv * np.array(dic1[0:int(input_dim1)])).reshape(1)
    # term2 = np.array(0.5 * np.array(dic1[int(input_dim1):int(2*input_dim1)]).T * np.matrix(np.eye(input_dim1)) * np.array(dic1[int(input_dim1):int(2*input_dim1)])).reshape(1)
    # H = term1 + term2

    #
    # ******** nD Even Rosenbrock #********
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = 0.0
    # for ii in np.arange(0,int(input_dim1/2),1):
    #     ind1 = ii
    #     ind2 = ii+1
    #     term1 = term1 + ((dic1[ind1] - 1.0)**2 - 100.0 * (dic1[ind2] - dic1[ind1]**2)**2) / 20.0 # (100 * (dic1[ii+1] - dic1[ii]**2)**2 + (1 - dic1[ii])**2) / 20.0
    # term2 = 0.0
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + 1*dic1[ii]**2/2
    # H = term1 + term2
    #
    # ******** 20D German Credit Data #********
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
    #
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = 0.0
    # param = dic1[0:input_dim1]
    # for ii in np.arange(0,1000,1):
    #     f_i = np.log(1+np.exp(np.sum(data[ii,1:21]*param)*data[ii,0])) + np.sum(param*param).astype(float)/(2000.0) #
    #     term1 = term1 + f_i
    # term2 = 0.0
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + 1*dic1[ii]**2/2
    # H = term1 + term2
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
    #
    #
    # dic1 = np.split(coords,2*input_dim1)
    # term1 = 0.0
    # param = dic1[0:input_dim1]
    # # for ii in np.arange(0,1000,1):
    # #     f_i = np.log(1+np.exp(np.sum(data[ii,1:21] * np.array(param).reshape(20))*data[ii,0])) + np.sum(np.array(param).reshape(20) * np.array(param).reshape(20))/(2000.0) #
    # #     term1 = term1 + f_i
    # term2 = 0.0
    # term1 = np.sum(np.log(np.exp(np.sum(data[:,1:21] * np.array(param).reshape(20),axis=1)*data[:,0])+1)+ np.sum(np.array(param).reshape(20) * np.array(param).reshape(20))/(2000.0))
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     term2 = term2 + 1*dic1[ii]**2/2
    # H = term1 + term2

    # ******** nD Gaussian with Wishart covariance #********
    dic1 = np.split(coords,2*input_dim1)
    sigma_inv = pickle.load(open("/Users/dhulls/Desktop/myfile.pkl", "rb"))
    sigma_inv = np.array(sigma_inv)
    term1 = 0.5 * np.matmul(np.matmul(np.array(dic1[0:int(input_dim1)]).T, sigma_inv), np.array(dic1[0:int(input_dim1)]))
    term2 = 0.5 * np.matmul(np.array(dic1[int(input_dim1):int(2*input_dim1)]).T, np.array(dic1[int(input_dim1):int(2*input_dim1)]))
    H = term1 + term2

    #******** Neural ODE #********
    
    # dic1 = np.split(coords,2*input_dim1)
    # init_params = convert_to_nn_tensor(dic1[0:int(input_dim1)], 5)
    # func = ODEFuncGrad(init_params).to(device)
    # pred_y = odeint(func.float(), batch_y0, batch_t).to(device)
    # loss = torch.mean(torch.abs(pred_y - batch_y))
    # term1 = loss.detach().numpy()
    
    # for ii in np.arange(0,2*input_dim1,1):
    #     term1 = term1 + 1*dic1[ii]**2/2
    # H = term1

    return H

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

def getgrad(coords):
    dic1 = np.split(coords,2*input_dim1)
    sigma_inv = pickle.load(open("/Users/dhulls/Desktop/myfile.pkl", "rb"))
    sigma_inv = np.matrix(sigma_inv)
    grad_req = np.zeros(2*input_dim1)
    grad_req[0:input_dim1] = np.array(sigma_inv * np.array(dic1[0:int(input_dim1)])).reshape(input_dim1)
    grad_req[int(input_dim1):int(2*input_dim1)] = np.array(dic1[int(input_dim1):int(2*input_dim1)]).reshape(input_dim1)
    return grad_req

def dynamics_fn(t, coords):
    # print("Here")
    # dcoords = autograd.grad(hamiltonian_fn)(coords) # func1
    dcoords = getgrad(coords)
    dic1 = np.split(dcoords,2*input_dim1)
    S = np.concatenate([dic1[input_dim1]])
    for ii in np.arange(input_dim1+1,2*input_dim1,1):
        S = np.concatenate([S, dic1[ii]])
    for ii in np.arange(0,input_dim1,1):
        S = np.concatenate([S, -dic1[ii]])
    return S

def get_trajectory(t_span=[0,250], timescale=40, radius=None, y0=None, noise_std=0.01, **kwargs): # 30 20
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))

    if y0 is None:
        y0 = np.zeros(input_dim1)
        for ii in np.arange(0,input_dim1,1):
            y0[ii] = norm(loc=0,scale=1).rvs()
    # spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs) #  method='RK45',
    lp_ivp = leapfrog(dynamics_fn, t_span, y0,int(timescale*(t_span[1]-t_span[0])), 2*input_dim1)
    # print(spring_ivp['y'])
    # print(lp_ivp)
    # q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = spring_ivp['y'][0], spring_ivp['y'][1], spring_ivp['y'][2], spring_ivp['y'][3], spring_ivp['y'][4], spring_ivp['y'][5], spring_ivp['y'][6], spring_ivp['y'][7], spring_ivp['y'][8], spring_ivp['y'][9], spring_ivp['y'][10], spring_ivp['y'][11], spring_ivp['y'][12], spring_ivp['y'][13], spring_ivp['y'][14], spring_ivp['y'][15], spring_ivp['y'][16], spring_ivp['y'][17], spring_ivp['y'][18], spring_ivp['y'][19]
    # dic1 = np.split(spring_ivp['y'], 2*input_dim1)
    dic1 = np.split(lp_ivp, 2*input_dim1)
    # dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = [dynamics_fn(None, lp_ivp[:,ii]) for ii in range(0, lp_ivp.shape[1])]
    dydt = np.stack(dydt).T
    # dq1dt, dq2dt, dq3dt, dq4dt, dq5dt, dq6dt, dq7dt, dq8dt, dq9dt, dq10dt, dp1dt, dp2dt, dp3dt, dp4dt, dp5dt, dp6dt, dp7dt, dp8dt, dp9dt, dp10dt = np.split(dydt,20)
    ddic1 = np.split(dydt, 2*input_dim1)

    # add noise
    # q += np.random.randn(*q.shape)*noise_std
    # p += np.random.randn(*p.shape)*noise_std
    return dic1, ddic1, t_eval

def get_dataset(seed=0, samples=Nsamps, test_split=1.0, **kwargs):
    
    data = from_pickle('/Users/dhulls/projects/Small Pf/hamiltonian-nn/nD_pdf/100D_Gaussian_1.pkl') # 100D_Gaussian.pkl
    print("Successfully loaded data")
    
    # data = {'meta': locals()}
    # # randomly sample inputs
    # np.random.seed(seed) #
    # xs, dxs = [], []
    # index1 = 0

    # count1 = 0
    # # y_init = np.random.rand(40)
    # # y_init = np.array([-0.73442741,  0.87541503, -0.91145534,  2.23884429, -0.94334989, -0.43981041, -1.32755931, -1.07763671, -0.76373273, -0.04889701, -0.85163158, -1.53461637, -0.05483824, -1.02400955,  0.20548304, -1.16507185, -1.10568765, -0.66630082, -1.32053015, -0.86211622,  1.13940068, -1.23482582,  0.40234164, -0.68481009, -0.87079715, -0.57884966, -0.31155253,  0.05616534, -1.16514984,  0.90082649, 0.46566244, -1.53624369,  1.48825219,  1.89588918,  1.17877957, -0.17992484, 1.07075262,  1.05445173, -0.40317695,  1.22244507])
    # y_init = np.zeros(2*input_dim1)
    # for ii in np.arange(0,input_dim1,1):
    #     y_init[ii] = 1.0 # lhd[count1,ii] # 0.0 # norm(loc=0,scale=1).rvs() #  lhd[count1,ii] # uniform(loc=-2,scale=4).rvs()
    # for ii in np.arange(input_dim1,2*input_dim1,1):
    #     y_init[ii] = norm(loc=0,scale=1).rvs() # lhd_p[count1,ii] #
    #     # if ii == input_dim1:
    #     #     y_init[ii] = norm(loc=0,scale=1).rvs()
    #     # else:
    #     #     y_init[ii] = norm(loc=0,scale=(2.718281828459045**(y_init[0] / 2))**(-1)).rvs()

    # for s in range(samples):
    #     print(s)
    #     dic1, ddic1, t = get_trajectory(y0=y_init,**kwargs) #
    #     # print(dic1)
    #     xs.append(np.stack( [dic1[ii].T.reshape(len(dic1[ii].T)) for ii in np.arange(0,2*input_dim1,1)]).T)
    #     dxs.append(np.stack( [ddic1[ii].T.reshape(len(ddic1[ii].T)) for ii in np.arange(0,2*input_dim1,1)]).T)
    #     y_init = np.zeros(2*input_dim1)
    #     count1 = count1 + 1
    #     for ii in np.arange(0,input_dim1,1):
    #         y_init[ii] = dic1[ii].T[len(dic1[ii].T)-1] #  lhd[count1,ii] #  uniform(loc=-2,scale=4).rvs() # lhd[count1,ii] #
    #     for ii in np.arange(input_dim1,2*input_dim1,1):
    #         y_init[ii] = norm(loc=0,scale=1).rvs() # lhd_p[count1,ii] #
    #         # if ii == input_dim1:
    #         #     y_init[ii] = norm(loc=0,scale=1).rvs()
    #         # else:
    #         #     y_init[ii] = norm(loc=0,scale=(2.718281828459045**(y_init[0] / 2))**(-1)).rvs()

    # data['coords'] = np.concatenate(xs)
    # data['dcoords'] = np.concatenate(dxs).squeeze()

    # # make a train/test split
    # split_ix = int(len(data['coords']) * test_split)
    # split_data = {}
    # for k in ['coords', 'dcoords']:
    #     split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    # data = split_data
    # path = '{}/{}.pkl'.format('/Users/dhulls/projects/Small Pf/hamiltonian-nn/nD_pdf', '100D_Gaussian_1')
    # to_pickle(data, path)
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
