import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('./')

from models.heat_cell import heat_cell

parser = argparse.ArgumentParser(description='Paths, hyperparameters and switches')
parser.add_argument('--dataset', default=None, help='Path to dataset')
parser.add_argument('--model_path', default=None, help='Path to saved model')
parser.add_argument('--param_path', default=None, help='Path to saved params')
parser.add_argument('--normalization', type=int, default=1, help='Normalize the data? 1 or 0')
args = parser.parse_args()
if args.normalization == 0:
    normalization = False
else:
    normalization = True


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Load and process dataset and make train-val splits
print("Loading dataset\n")
obs_maps = np.load(args.dataset)
print("Preparing data for training\n")
if normalization:
    omin, omax = np.min(obs_maps), np.max(obs_maps)
    obs_maps = (obs_maps - omin) / (omax - omin)
num_samples, num_frames, H, W = obs_maps.shape

T0 = obs_maps[:, :-2].reshape(-1, 1, H, W)
T1 = obs_maps[:, 1:-1].reshape(-1, 1, H, W)
T2 = obs_maps[:, 2:].reshape(-1, 1, H, W)

obs_maps = np.concatenate([T0, T1, T2], axis=1)
val_split = 0.1
train_obs_maps = obs_maps[int(val_split*len(obs_maps)):]
val_obs_maps = obs_maps[:int(val_split*len(obs_maps))]
np.random.shuffle(train_obs_maps)
train_obs_maps = torch.from_numpy(train_obs_maps).float()
val_obs_maps = torch.from_numpy(val_obs_maps).float()
if torch.cuda.is_available():
    train_obs_maps = train_obs_maps.cuda()
    val_obs_maps = val_obs_maps.cuda()

# Define network
net = heat_cell()
if torch.cuda.is_available():
    net = net.cuda()
# Load any pretrained model if available
if args.model_path is not None:
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint)
    checkpoint = torch.load(args.param_path)
    net.alpha = checkpoint['alpha']
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
                       {'params': [net.alpha, net.sf], 'lr': 1e-3}
                       ], lr=1e-3) 

#optimizer = optim.Adam(net.parameters(), lr=1e-3)

criterion = nn.MSELoss()

lambd = 0e-5

# Train
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
        y = train_obs_maps[step*batch_size : (step+1)*batch_size, 2]
        # Forward
        y_pred, V, V_hat = iter_cell(net, X)
        # Compute loss
        train_loss = criterion(y_pred[-2], y) + criterion(V_hat[-2], V[-1])
        # Add sparsity loss if normalization is disabled
        if not normalization:
            train_loss += lambd * torch.mean(torch.abs(V_hat[-2])) 
        # Backprop
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if net.alpha < 0 or net.sf < 0:
           print("Re-initializing diffusion coefficient")
           nn.init.uniform_(net.alpha, 0, 1.)
           nn.init.uniform_(net.sf, 0, 0.01)
        train_error += train_loss.item()
        
    # Validation
    with torch.no_grad():
        val_error = 0.
        for vstep in range(val_steps):
            val_X = val_obs_maps[vstep*batch_size : (vstep+1)*batch_size]
            val_y = val_obs_maps[vstep*batch_size : (vstep+1)*batch_size, 2]
            val_y_pred, _, _ = iter_cell(net, val_X)
            val_loss = criterion(val_y_pred[-2], val_y)
            val_error += val_loss

    print("Epoch: {}, Training Error: {}, Validation Error: {}".format(epoch+1,
                                                                      train_error/train_steps,
                                                                      val_error/val_steps))

    # Exponential decay learning rate 
    for g in optimizer.param_groups:
        if g['lr'] > 0.0001:
            g['lr'] *= 0.99
        print("LR: {}".format(g['lr']))

    # Save models
    torch.save(net.state_dict(), 'saved_models/heat_system/model-{}.ckpt'.format(epoch))
    dict = {}
    dict['alpha'] = net.alpha
    dict['sf'] = net.sf
    torch.save(dict, 'saved_models/heat_system/parameters-{}.ckpt'.format(epoch))


 
