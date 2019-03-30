# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse, os
import numpy as np

from nn_models import MLPAutoencoder, MLP
from hnn import HNN, HNNBaseline, PixelHNN
from gym_dataloader import get_dataset
from utils import L2_loss

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=784, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--latent_dim', default=2, type=int, help='latent dimension of autoencoder')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=200, type=int, help='batch size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=3000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=250, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='pendulum', type=str, help='either "real" or "sim" data')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default='./saved', type=str, help='name of dataset')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  autoencoder = MLPAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim,
                               nonlinearity='relu')
  model = PixelHNN(args.latent_dim, args.hidden_dim,
                   autoencoder=autoencoder, nonlinearity=args.nonlinearity,
                   baseline=args.baseline)
  optim = torch.optim.Adam(model.parameters(), args.learn_rate)

  # get dataset
  data = get_dataset(args.name, args.save_dir, verbose=True, seed=args.seed)
  trials = data['meta']['trials']
  timesteps = data['meta']['timesteps']
  inputs = torch.tensor( data['pixels'], dtype=torch.float32)
  inputs_next = torch.tensor( data['next_pixels'], dtype=torch.float32)

  # vanilla ae train loop
  for step in range(args.total_steps+1):
    # select a batch
    ixs = torch.randperm(trials*timesteps-2*trials)[:args.batch_size]
    x = inputs[ixs]
    x_next = inputs_next[ixs]

    # encode pixel space -> latent dimension
    z = model.encode(x)
    z_next = model.encode(x_next)

    # autoencoder loss
    x_hat = model.decode(z)
    ae_loss = L2_loss(x, x_hat)

    # hnn vector field loss
    z_hat_next = z + model.time_derivative(z)
    hnn_loss = L2_loss(z_next, z_hat_next)

    # canonical coordinate loss
    # -> makes latent space look like (x, v) coordinates
    w, dw = z.split(1,1)
    w_next, _ = z_next.split(1,1)
    cc_loss = L2_loss(dw, w_next - w)

    # sum losses and take a gradient step
    loss = cc_loss + ae_loss + 1e-2 * hnn_loss
    loss.backward() ; optim.step() ; optim.zero_grad()

    if step % 250 == 0:
      print("step {}, loss {:.4e}".format(step, loss.item()))

  return model

if __name__ == "__main__":
    args = get_args()
    model = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = 'baseline' if args.baseline else 'hnn'
    path = '{}/{}-pixel-{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)