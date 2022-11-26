import torch
import torch.nn as nn

from models.RedNet import RedNet

class wave_cell(nn.Module):

    def __init__(self, slen=4):
        super(wave_cell, self).__init__()
        self.slen = slen
        self.c2 = torch.empty(1, requires_grad=True) # wave speed
        nn.init.uniform_(self.c2, 0, 1.)
        self.sf = torch.empty(1, requires_grad=True) # scale factor for source map
        nn.init.normal_(self.sf, 0, 0.01)
        self.spatial_kernel = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.spatial = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.spatial.weight = nn.Parameter(self.spatial_kernel.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.Wcc_u = torch.Tensor([[0, 0], [1, 0]])   # 2x2 as temporal order = 2
        self.Wcc_u.requires_grad = False
        self.wcu = torch.Tensor([1, 0])
        self.wcu.requires_grad = False
        self.whc = torch.Tensor([2, -1])
        self.whc.requires_grad = False
        self.Wcc_v = torch.zeros(slen, slen)
        for j in range(slen - 1):
            self.Wcc_v[j + 1, j] = 1 
        self.Wcc_v.requires_grad = False
        self.wcv = torch.zeros(slen)
        self.wcv[0] = 1
        self.wcv.requires_grad = False

        self.RedNet = RedNet(in_channels=slen, num_blocks=3, hidden_channels=32, positive=False)

    def forward(self, inputs, init_states=None):
        batch_size, height, width = inputs.shape

        if init_states is None:
            H = torch.zeros(batch_size, height, width)
            C = torch.zeros(batch_size, 2 + self.slen, height, width)  # temporal order = 2
        else:
            H, C = init_states

        # Estimate source map
        V = inputs - H

        # Compute cell state
        Cu, Cv = C[:, :2], C[:, 2:]

        Cu = Cu.permute(0, 3, 2, 1)
        Wcc_u = torch.transpose(self.Wcc_u, 0, 1)
        Wcc_u = Wcc_u.repeat(batch_size, width, 1, 1)
        Cu = torch.matmul(Cu, Wcc_u)
        Cu = Cu.permute(0, 3, 2, 1)
        wcu = self.wcu.repeat(batch_size, 1)
        wcu = wcu.unsqueeze(-1).unsqueeze(-1)
        U = inputs.unsqueeze(1)
        U = U.repeat(1, 2, 1, 1)  # index 1 is temporal order
        Cu = Cu + wcu * U

        Cv = Cv.permute(0, 3, 2, 1)
        Wcc_v = torch.transpose(self.Wcc_v, 0, 1)
        Wcc_v = Wcc_v.repeat(batch_size, width, 1, 1)
        Cv = torch.matmul(Cv, Wcc_v)
        Cv = Cv.permute(0, 3, 2, 1)
        wcv = self.wcv.repeat(batch_size, 1)
        wcv = wcv.unsqueeze(-1).unsqueeze(-1)
        Vs = V.unsqueeze(1)
        Vs = Vs.repeat(1, self.slen, 1, 1)  # index 1 is temporal order
        Cv = Cv + wcv * Vs

        C = torch.cat([Cu, Cv], dim=1)

        # Compute homogeneous solution
        whc = self.whc.repeat(batch_size, width, 1)
        whc = whc.unsqueeze(-1)
        H = torch.matmul(Cu.permute(0, 3, 2, 1), whc)
        H = H.permute(3, 0, 2, 1)[0]
        H = H + self.c2 * self.spatial(inputs.unsqueeze(1))[:, 0]

        # Predict next source map
        V_hat = Cv / self.sf
        V_hat = self.RedNet(V_hat)
        V_hat = V_hat[:, 0]
        V_hat = V_hat * self.sf

        # Compute output
        outputs = H + V_hat

        return outputs, H, C, V, V_hat
