# Sam Greydanus, Misko Dzama, Jason Yosinski
# 2019 | Google AI Residency Project "Hamiltonian Neural Networks"

import numpy as np
import os, torch, pickle, zipfile
import imageio, shutil
import scipy, scipy.misc, scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
# from data import hamiltonian_fn


# def integrate_model(model, t_span, y0, fun=None, **kwargs):
#   def default_fun(t, np_x):
#       x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
#       x = x.view(1, np.size(np_x)) # batch size of 1
#       dx = model.time_derivative(x).data.numpy().reshape(-1)
#       return dx
#   fun = default_fun if fun is None else fun
#   return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

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

def integrate_model(model, t_span, y0, n, **kwargs):
  def default_fun(t, np_x):
      x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
      x = x.view(1, np.size(np_x)) # batch size of 1
      dx = model.time_derivative(x).data.numpy().reshape(-1)
      return dx
  fun = default_fun # if fun is None else fun
  return leapfrog(fun, t_span, y0, n, 10)
  # return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

# def rk4(fun, y0, t, dt, *args, **kwargs):
#   dt2 = dt / 2.0
#   k1 = fun(y0, t, *args, **kwargs)
#   k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
#   k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
#   k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
#   dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
#   return dy

def rk4(fun, y0, t, dt, *args, **kwargs):
  k1 = fun(y0, t-dt, *args, **kwargs)
  k2 = fun(y0, t+dt, *args, **kwargs)
  dy = (k2-k1) / (2*dt)
  return dy


# def LH_loss(model, x, step):
#     diff = 0
#     base_H = hamiltonian_fn(x[0]).detach().numpy()
#     # print(x.shape)
#     N = np.min([x.shape[0]/20, 10])
#     # Nrnd = 20
#     # hnn_ivp = integrate_model(model, [0, 10], x[0].detach().numpy())
#     count = 0
#     kwargs = {'t_eval': np.linspace(0, 10, 200), 'rtol': 1e-5}
#     steps = 200
#     for ii in np.arange(0,N,1):
#         idx = int(ii*20-1)
#         hnn_ivp = integrate_model(model, [0, 10], x[idx].detach().numpy(), steps, **kwargs)
#         # rnd_ind = torch.randperm(1600)
#         # diff = diff + (hamiltonian_fn(hnn_ivp.y[:,79]) - base_H)**2
#         diff = diff + (hamiltonian_fn(hnn_ivp[:,199]) - base_H)**2
#         count = count + 1
#         # print(str(count) + "   " + str(step))
#         # for jj in np.arange(0,Nrnd,1): # hnn_ivp.y.shape[1]
#         #     diff = diff + (hamiltonian_fn(hnn_ivp.y[:,rnd_ind[jj]]) - base_H)**2
#         #     count = count + 1
#     return (diff / count)

def L2_loss(u, v):
  return (u-v).pow(2).mean()


def read_lipson(experiment_name, save_dir):
  desired_file = experiment_name + ".txt"
  with zipfile.ZipFile('{}/invar_datasets.zip'.format(save_dir)) as z:
    for filename in z.namelist():
      if desired_file == filename and not os.path.isdir(filename):
        with z.open(filename) as f:
            data = f.read()
  return str(data)


def str2array(string):
  lines = string.split('\\n')
  names = lines[0].strip("b'% \\r").split(' ')
  dnames = ['d' + n for n in names]
  names = ['trial', 't'] + names + dnames
  data = [[float(s) for s in l.strip("' \\r,").split( )] for l in lines[1:-1]]

  return np.asarray(data), names


def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


def choose_nonlinearity(name):
  nl = None
  if name == 'tanh':
    nl = torch.tanh
  elif name == 'relu':
    nl = torch.relu
  elif name == 'sigmoid':
    nl = torch.sigmoid
  elif name == 'softplus':
    nl = torch.nn.functional.softplus
  elif name == 'selu':
    nl = torch.nn.functional.selu
  elif name == 'elu':
    nl = torch.nn.functional.elu
  elif name == 'swish':
    nl = lambda x: x * torch.sigmoid(x)
  elif name == 'sine':
    nl = lambda x: torch.sin(x)
  else:
    raise ValueError("nonlinearity not recognized")
  return nl


def make_gif(frames, save_dir, name='pendulum', duration=1e-1, pixels=None, divider=0):
    '''Given a three dimensional array [frames, height, width], make
    a gif and save it.'''
    temp_dir = './_temp'
    os.mkdir(temp_dir) if not os.path.exists(temp_dir) else None
    for i in range(len(frames)):
        im = (frames[i].clip(-.5,.5) + .5)*255
        im[divider,:] = 0
        im[divider + 1,:] = 255
        if pixels is not None:
          im = scipy.misc.imresize(im, pixels)
        scipy.misc.imsave(temp_dir + '/f_{:04d}.png'.format(i), im)

    images = []
    for file_name in sorted(os.listdir(temp_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(temp_dir, file_name)
            images.append(imageio.imread(file_path))
    save_path = '{}/{}.gif'.format(save_dir, name)
    png_save_path = '{}.png'.format(save_path)
    imageio.mimsave(save_path, images, duration=duration)
    os.rename(save_path, png_save_path)

    shutil.rmtree(temp_dir) # remove all the images
    return png_save_path
