import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import sys
sys.path.append('./')

from models.wave_cell import wave_cell

parser = argparse.ArgumentParser(description='Paths and switches')
parser.add_argument('--dataset', default='./dataset/test_wave_maps.npy', help='Path to wavemaps')
parser.add_argument('--model_path', default=None, help='Path to saved models')
parser.add_argument('--param_path', default=None, help='Path to saved parameters')
parser.add_argument('--seqNo', default=0, help='Seq No.')
parser.add_argument('--savepath', default=None, help='Path to save results')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

slen = 2

# Load and process dataset
obs_maps = np.load(args.dataset)
omin, omax = -0.07274206, 0.397301 # minimum and maximum value in training dataset

obs_maps = (obs_maps - omin) / (omax - omin)
num_samples, num_frames, H, W = obs_maps.shape

obs_maps = obs_maps[int(args.seqNo)]
if torch.cuda.is_available():
    obs_maps = torch.from_numpy(obs_maps).float().cuda()
else:
    obs_maps = torch.from_numpy(obs_maps).float()

# Define network
net = wave_cell(slen=slen)
if torch.cuda.is_available():
    net = net.cuda()
# Load weights from trained model
checkpoint = torch.load(args.model_path)
net.load_state_dict(checkpoint)
checkpoint = torch.load(args.param_path)
net.c2 = checkpoint['c2']
net.sf = checkpoint['sf']


net = net.eval()

# Create output paths
if not os.path.exists(args.savepath):
    os.makedirs(os.path.join(args.savepath, 'snr'))
    os.makedirs(os.path.join(args.savepath, 'gt'))
    os.makedirs(os.path.join(args.savepath, 'pred'))
error_file = open(os.path.join(args.savepath, 'snr', 'seq_{}.txt'.format(args.seqNo)), 'w')

# Define loss/error function
criterion = nn.MSELoss()

# Initial observation
X = obs_maps[0:1]
y = obs_maps[1:]

# Iterative prediction
y_pred = torch.zeros_like(y)
temporal_order = 2
for t in range(199):
    print(t)
    with torch.no_grad():
        if t == 0:
            pred, Ht, Ct, Vt, Vt_hat = net(X)
        else:
            pred, Ht, Ct, Vt, Vt_hat = net(X, (Ht, Ct))
    y_pred[t] = pred
    if t < temporal_order + slen - 1:
        X = obs_maps[t+1:t+2]
    else:
        X = pred

        
y = y * (omax - omin) + omin
y_pred = y_pred * (omax - omin) + omin

# Save SNR and visual maps
for t in range(temporal_order + slen - 1, 199):
    snr = torch.norm(y[t])/torch.dist(y[t], y_pred[t])
    snr = 20 * torch.log10(snr)
    print('Frame: {}, SNR: {}'.format(t, snr))

    error_file.write('Frame: {}, SNR: {}'.format(t, snr))
    error_file.write("\n")

    plt.imshow(y[t].cpu().numpy(), cmap='seismic')
    plt.axis('off')
    plt.savefig(os.path.join(args.savepath, 'gt/{0:05d}.png'.format(t)))
    plt.close()
    plt.imshow(y_pred[t].cpu().numpy(), cmap='seismic')
    plt.axis('off')
    plt.savefig(os.path.join(args.savepath, 'pred/{0:05d}.png'.format(t)))
    plt.close()
