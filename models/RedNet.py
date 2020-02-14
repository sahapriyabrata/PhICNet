import torch
import torch.nn as nn


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

class conv_block(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, stride=1, init_weights=True):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        if init_weights:
            initialize_weights(self)        

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        
        return x 

class t_conv_block(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, stride=1, init_weights=True):
        super(t_conv_block, self).__init__()
        self.tconv1 = nn.ConvTranspose2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.tconv2 = nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=stride-1)
        if init_weights:
            initialize_weights(self)

    def forward(self, x):
        x = self.tconv1(x)
        x = torch.relu(x)
        x = self.tconv2(x)

        return x

class RedNet(nn.Module):
    def __init__(self, num_blocks=3, in_channels=1,  out_channels=1, positive=True):
        super(RedNet, self).__init__()
        self.positive = positive
        if num_blocks == 3:
            self.conv_blocks = nn.ModuleList([conv_block(in_channels=in_channels, out_channels=32, stride=2)])
            self.conv_blocks.append(conv_block(in_channels=32, out_channels=32, stride=2))
            self.conv_blocks.append(conv_block(in_channels=32, out_channels=32, stride=2))
            self.t_conv_blocks = nn.ModuleList([t_conv_block(in_channels=32, out_channels=32, stride=2)])
            self.t_conv_blocks.append(t_conv_block(in_channels=32, out_channels=32, stride=2))
            self.t_conv_blocks.append(t_conv_block(in_channels=32, out_channels=out_channels, stride=2))
        elif num_blocks == 5:
            self.conv_blocks = nn.ModuleList([conv_block(in_channels=in_channels, out_channels=32, stride=2)])
            self.conv_blocks.append(conv_block(in_channels=32, out_channels=32, stride=1))
            self.conv_blocks.append(conv_block(in_channels=32, out_channels=32, stride=2))
            self.conv_blocks.append(conv_block(in_channels=32, out_channels=32, stride=1))
            self.conv_blocks.append(conv_block(in_channels=32, out_channels=32, stride=2))
            self.t_conv_blocks = nn.ModuleList([t_conv_block(in_channels=32, out_channels=32, stride=2)])
            self.t_conv_blocks.append(t_conv_block(in_channels=32, out_channels=32, stride=1))
            self.t_conv_blocks.append(t_conv_block(in_channels=32, out_channels=32, stride=2))
            self.t_conv_blocks.append(t_conv_block(in_channels=32, out_channels=32, stride=1))
            self.t_conv_blocks.append(t_conv_block(in_channels=32, out_channels=out_channels, stride=2))            
            

    def forward(self, inputs):
        x = inputs
        encs = [x[:, -1:]]
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x)
            encs.append(x)
        for j in range(len(self.t_conv_blocks) - 1):
            x = self.t_conv_blocks[j](x)
            x = torch.relu(x + encs[i - j])
        x = self.t_conv_blocks[-1](x) + encs[0]
        if self.positive:
            x = torch.relu(x)

        return x
                    
