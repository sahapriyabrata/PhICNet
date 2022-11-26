import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('./')

from models.wave_cell import wave_cell
from utils import numpy2torch

parser = argparse.ArgumentParser(description='Paths')
parser.add_argument('--dataset', default=None, help='Path to dataset')
parser.add_argument('--model_path', default=None, help='Path to pretrained model')
parser.add_argument('--param_path', default=None, help='Path to pretrained params')
parser.add_argument('--savepath', default=None, help='Path to save models')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

slen = 2

# Load and process dataset and make train-val splits
print("Loading dataset\n")
obs_maps = np.load(args.dataset)#[:300]
print("Preparing data for training\n")
omin, omax = np.min(obs_maps), np.max(obs_maps)
print(omin, omax)
obs_maps = (obs_maps - omin) / (omax - omin)
num_samples, num_frames, H, W = obs_maps.shape

seqs = []
for i in range(2 + slen): # temporal order of homogeneous PDE is 2
    seqs.append(obs_maps[:, i:-(2 + slen - i)].reshape(-1, H, W))
seqs.append(obs_maps[:, 2 + slen:].reshape(-1, H, W))
seqs = np.array(seqs)
obs_maps = np.swapaxes(seqs, 0, 1)

val_split = 0.1
train_obs_maps = obs_maps[int(val_split*len(obs_maps)):]
val_obs_maps = obs_maps[:int(val_split*len(obs_maps))]
np.random.shuffle(train_obs_maps)

# Define network
net = wave_cell(slen=slen)
if torch.cuda.is_available():
    net = net.cuda()
# Load any pretrained model if available
if args.model_path is not None:
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint)
    checkpoint = torch.load(args.param_path)
    net.c2 = checkpoint['c2']
    net.sf = checkpoint['sf']
net = net.train()

# Unrolling PhICNet cell for given sequence length
def iter_cell(net, X):
    y = []
    V = []
    V_hat = []
    for t in range(X.shape[1]):
        if t == 0:
            yt, Ht, Ct, Vt, Vt_hat = net(X[:, t])
        else:
            yt, Ht, Ct, Vt, Vt_hat = net(X[:, t], (Ht, Ct))
        y.append(yt)
        V.append(Vt)
        V_hat.append(Vt_hat)

    return  y, V, V_hat


# Define optimizer and loss
optimizer = optim.Adam([
                       {'params': net.parameters()},
                       {'params': [net.c2, net.sf], 'lr': 1e-3}
                       ], lr=1e-3)

#optimizer = optim.Adam(net.parameters(), lr=1e-3)

criterion = nn.MSELoss()

lambd = 1e-7

# Train
writer = SummaryWriter()

num_epochs = 200
batch_size = 32
train_steps = len(train_obs_maps)//batch_size
val_steps = len(val_obs_maps)//batch_size
print("Starting training\n")
for epoch in range(num_epochs):
    train_error = 0.
    for step in range(train_steps):
        # Get batch data
        X = train_obs_maps[step*batch_size : (step+1)*batch_size]
        y = train_obs_maps[step*batch_size : (step+1)*batch_size, -1]
        X = numpy2torch(X)
        y = numpy2torch(y)
        # Forward
        y_pred, V, V_hat = iter_cell(net, X)
        # Compute loss
        train_loss = criterion(y_pred[-2], y) + criterion(V_hat[-2], V[-1]) + lambd * torch.mean(torch.abs(V_hat[-2]))
        # Backprop
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if net.c2 < 0 or net.sf < 0: # or net.sf[1] < 0:
           print("Re-initializing parameters")
           nn.init.uniform_(net.c2, 0, 1.)
           nn.init.uniform_(net.sf, 0, 0.01)
        train_error += train_loss.item()

    
    # Validation
    with torch.no_grad():
        val_error = 0.
        for vstep in range(val_steps):
            val_X = val_obs_maps[vstep*batch_size : (vstep+1)*batch_size]
            val_y = val_obs_maps[vstep*batch_size : (vstep+1)*batch_size, -1]
            val_X = numpy2torch(val_X)
            val_y = numpy2torch(val_y)
            val_y_pred, _, _ = iter_cell(net, val_X)
            val_loss = criterion(val_y_pred[-2], val_y)
            val_error += val_loss

    print("Epoch: {}, Training Error: {}, Validation Error: {}".format(epoch,
                                                                       train_error/train_steps,
                                                                       val_error/val_steps))

    # Exponential decay learning rate 
    for g in optimizer.param_groups:
        if g['lr'] > 0.0001:
            g['lr'] *= 0.99
        print("LR: {}".format(g['lr']))

    # Write to tensorboard
    writer.add_scalar('Loss/train/wave', train_error/train_steps, epoch)
    writer.add_scalar('Loss/val/wave', val_error/val_steps, epoch)

    # Save models
    if not os.path.exists(args.savepath):
        os.makedirs(os.path.join(args.savepath))
    torch.save(net.state_dict(), os.path.join(args.savepath, 'model-{0:04d}.ckpt'.format(epoch)))
    dict = {}
    dict['c2'] = net.c2
    dict['sf'] = net.sf
    torch.save(dict, os.path.join(args.savepath, 'parameters-{0:04d}.ckpt'.format(epoch))) 
