# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from hnn import HNN
from data import get_dataset # , hamiltonian_fn
from utils import L2_loss, rk4, integrate_model # LH_loss,

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=6, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=100, type=int, help='hidden dimension of mlp') # 100
    # parser.add_argument('--t_span', default=4, type=int, help='t_span')
    # parser.add_argument('--timescale', default=400, type=int, help='timescale')
    parser.add_argument('--learn_rate', default=5e-4, type=float, help='learning rate') #
    parser.add_argument('--batch_size', default=1000, type=int, help='batch_size') # 2000
    parser.add_argument('--input_noise', default=0.0, type=int, help='std of noise added to inputs')
    parser.add_argument('--nonlinearity', default='sine', type=str, help='neural net nonlinearity') # relu tanh
    parser.add_argument('--total_steps', default=100000, type=int, help='number of gradient steps') # 25000
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='ndpdf', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  output_dim = args.input_dim if args.baseline else args.input_dim
  nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
  model = HNN(args.input_dim, differentiable_model=nn_model,
            field_type=args.field_type, baseline=args.baseline)
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0.)

  # arrange data
  data = get_dataset(seed=args.seed)
  x = torch.tensor( data['coords'], requires_grad=True, dtype=torch.float32)
  # tmp_mean = torch.mean(x, 0)
  # tmp_std = torch.std(x, 0)

  # x = torch.tensor(torch.sub(x,torch.mean(x, 0))).clone().detach().requires_grad_(True)
  # x = torch.tensor(torch.div(x,torch.std(x, 0))).clone().detach().requires_grad_(True)
  # x = torch.tensor(x, requires_grad=True, dtype=torch.float32)

  test_x = torch.tensor( data['test_coords'], requires_grad=True, dtype=torch.float32)
  dxdt = torch.Tensor(data['dcoords'])
  test_dxdt = torch.Tensor(data['test_dcoords'])

  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  # print(hamiltonian_fn(x[0])) # hamiltonian_fn(x)

  # print(hamiltonian_fn(torch.cat((f1[0],f2[0]),0)))
  # print(f1.size()[0])
  for step in range(args.total_steps+1):

    # train step (no batch)
    # f1, f2 = model.forward(x)
    # dxdt_hat = model.rk4_time_derivative(x) if args.use_rk4 else model.time_derivative(x)
    # loss = L2_loss(dxdt, dxdt_hat) + 0.1*LH_loss(f1, f2, x)
    # loss.backward() ; optim.step() ; optim.zero_grad()

    # train step (batch)
    ixs = torch.randperm(x.shape[0])[:args.batch_size]
    # f1, f2 = model.forward(x[ixs]) #
    # hnn_ivp = integrate_model(model, [0, 4], x[0].detach().numpy())
    # print(hnn_ivp.y[0:args.input_dim,:].shape)
    # print(hamiltonian_fn(hnn_ivp.y[0:args.input_dim,0]))
    # print(LH_loss(model, x))
    dxdt_hat = model.rk4_time_derivative(x[ixs]) if args.use_rk4 else model.time_derivative(x[ixs])
    loss = L2_loss(dxdt[ixs], dxdt_hat) # + 1.0*torch.tensor( LH_loss(model, x, step), requires_grad=True, dtype=torch.float32)
    loss.backward() ; optim.step() ; optim.zero_grad()

    # ixs = torch.randperm(x.shape[0])[:args.batch_size]
    # dxdt_hat = model.time_derivative(x[ixs])
    # dxdt_hat += args.input_noise * torch.randn(*x[ixs].shape) # add noise, maybe
    # loss = L2_loss(dxdt[ixs], dxdt_hat)
    # loss.backward()
    # grad = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
    # optim.step() ; optim.zero_grad()

    # run test data
    test_dxdt_hat = model.rk4_time_derivative(test_x) if args.use_rk4 else model.time_derivative(test_x)
    test_loss = L2_loss(test_dxdt, test_dxdt_hat)
    # test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
    # test_dxdt_hat = model.time_derivative(test_x[test_ixs])
    # test_dxdt_hat += args.input_noise * torch.randn(*test_x[test_ixs].shape) # add noise, maybe
    # test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)

    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
    # if args.verbose and step % args.print_every == 0:
    #   print("step {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}"
    #       .format(step, loss.item(), test_loss.item(), grad@grad, grad.std()))

  train_dxdt_hat = model.time_derivative(x)
  # f1, f2 = model.forward(x)
  # print(torch.tensor( LH_loss(model, x, step), requires_grad=True, dtype=torch.float32))
  train_dist = (dxdt - train_dxdt_hat)**2  # + 1.0*torch.tensor( LH_loss(model, x, 0), requires_grad=True, dtype=torch.float32)
  test_dxdt_hat = model.time_derivative(test_x)
  test_dist = (test_dxdt - test_dxdt_hat)**2
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))
  # torch.set_printoptions(profile="full")
  # print(tmp_mean)
  # print(tmp_std)
  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-baseline' if args.baseline else '-hnn'
    label = '-rk4' + label if args.use_rk4 else label
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)
