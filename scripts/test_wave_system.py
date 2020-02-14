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
parser.add_argument('--normalization', type=int, default=1, help='Is the model trained with normalized data? 1 or 0')
args = parser.parse_args()
if args.normalization == 0:
    normalization = False
else:
    normalization = True


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Load dataset
obs_maps = np.load(args.dataset)
if normalization:
    omin, omax = -0.07274206, 0.397301 # minimum and maximum value in training dataset
    obs_maps = (obs_maps - omin) / (omax - omin)
num_samples, num_frames, H, W = obs_maps.shape

obs_maps = obs_maps[int(args.seqNo)]
if torch.cuda.is_available():
    obs_maps = torch.from_numpy(obs_maps).float().cuda()
else:
    obs_maps = torch.from_numpy(obs_maps).float()

# Define network
net = wave_cell()
if torch.cuda.is_available():
    net = net.cuda()
checkpoint = torch.load(args.model_path)
net.load_state_dict(checkpoint)
checkpoint = torch.load(args.param_path)
net.c2 = checkpoint['c2']
net.sf = checkpoint['sf']

net = net.eval()

if not os.path.exists('./results/wave_system/'):
    os.makedirs('./results/wave_system/snr/')
    os.makedirs('./results/wave_system/gt/')
    os.makedirs('./results/wave_system/pred/')
error_file = open(os.path.join('./results/wave_system/snr/') + 'snr_{}.txt'.format(args.seqNo), 'w')

criterion = nn.MSELoss()

X = obs_maps[0:1]
y = obs_maps[1:]

y_pred = torch.zeros_like(y)
temporal_order = 2
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

if normalization:
    y = y * (omax - omin) + omin
    y_pred = y_pred * (omax - omin) + omin

for t in range(temporal_order, 199):
    snr = torch.norm(y[t])/torch.dist(y[t], y_pred[t])
    snr = 20 * torch.log10(snr)
    print('Frame: {}, SNR: {}'.format(t, snr))

    error_file.write('Frame: {}, SNR: {}'.format(t, snr))
    error_file.write("\n")

    plt.imshow(y[t].cpu().numpy(), cmap='seismic')
    plt.axis('off')
    plt.savefig('./results/wave_system/gt/{}.png'.format(t))
    plt.close()
    plt.imshow(y_pred[t].cpu().numpy(), cmap='seismic')
    plt.axis('off')
    plt.savefig('./results/wave_system/pred/{}.png'.format(t))
    plt.close()

