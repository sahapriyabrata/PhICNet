import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import sys
sys.path.append('./')

from models.heat_cell import heat_cell

parser = argparse.ArgumentParser(description='Paths and switches')
parser.add_argument('--dataset', default='./dataset/test_heat_maps.npy', help='Path to heatmaps')
parser.add_argument('--model_path', default=None, help='Path to saved models')
parser.add_argument('--param_path', default=None, help='Path to saved parameters')
parser.add_argument('--seqNo', default=0, help='Seq No.')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Load and process dataset
obs_maps = np.load(args.dataset)
omin, omax = 2.8593703e-07, 0.23825705 # minimum and maximum value in training dataset
obs_maps = (obs_maps - omin) / (omax - omin)
num_samples, num_frames, H, W = obs_maps.shape

obs_maps = obs_maps[int(args.seqNo)]
if torch.cuda.is_available():
    obs_maps = torch.from_numpy(obs_maps).float().cuda()
else:
    obs_maps = torch.from_numpy(obs_maps).float()

# Define network
net = heat_cell()
if torch.cuda.is_available():
    net = net.cuda()
# Load weights from trained model
checkpoint = torch.load(args.model_path)
net.load_state_dict(checkpoint)
checkpoint = torch.load(args.param_path)
net.alpha = checkpoint['alpha']
net.sf = checkpoint['sf']

net = net.eval()

# Create output paths
if not os.path.exists('./results/heat_system/'):
    os.makedirs('./results/heat_system/snr/')
    os.makedirs('./results/heat_system/gt/')
    os.makedirs('./results/heat_system/pred/')
error_file = open(os.path.join('./results/heat_system/snr/') + 'snr_{}.txt'.format(args.seqNo), 'w')

# Define loss/error function
criterion = nn.MSELoss()

# Initial observation
X = obs_maps[0:1]
y = obs_maps[1:]

# Iterative prediction
y_pred = torch.zeros_like(y)
temporal_order = 1
for t in range(199):
    print(t)
    with torch.no_grad():
        if t == 0:
            pred, Ht, Ct, _, _ = net(X)
        else:
            pred, Ht, Ct, _, _ = net(X, (Ht, Ct))
    y_pred[t] = pred
    if t < temporal_order:
        X = obs_maps[t+1:t+2]
    else:
        X = pred
        
y = y * (omax - omin) + omin
y_pred = y_pred * (omax - omin) + omin

# Save SNR and visual maps
for t in range(temporal_order, 199):
    snr = torch.norm(y[t])/torch.dist(y[t], y_pred[t])
    snr = 20 * torch.log10(snr)
    print('Frame: {}, SNR: {}'.format(t, snr))

    error_file.write('Frame: {}, SNR: {}'.format(t, snr))
    error_file.write("\n")

    plt.imshow(y[t].cpu().numpy(), cmap='hot')
    plt.axis('off')
    plt.savefig('./results/heat_system/gt/{}.png'.format(t))
    plt.close()
    plt.imshow(y_pred[t].cpu().numpy(), cmap='hot')
    plt.axis('off')
    plt.savefig('./results/heat_system/pred/{}.png'.format(t))
    plt.close()

