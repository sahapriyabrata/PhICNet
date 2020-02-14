import torch
import torch.nn as nn

from models.RedNet import RedNet

class wave_cell(nn.Module):

    def __init__(self):
        super(wave_cell, self).__init__()
        self.c2 = torch.empty(1, requires_grad=True) # wave speed
        nn.init.uniform_(self.c2, 0, 1.)
        self.sf = torch.empty(1, requires_grad=True) # scale factor for source map
        nn.init.normal_(self.sf, 0, 0.01)
        self.spatial_kernel = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.spatial = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.spatial.weight = nn.Parameter(self.spatial_kernel.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.Wcc = torch.Tensor([[0, 0], [1, 0]])   # 2x2 as temporal order = 2
        self.Wcc.requires_grad = False
        self.wcu = torch.Tensor([1, 0])
        self.wcu.requires_grad = False
        self.whc = torch.Tensor([2, -1])
        self.whc.requires_grad = False
        self.RedNet = RedNet()

    def forward(self, inputs, init_states=None):
        batch_size, height, width = inputs.shape

        if init_states is None:
            H = torch.zeros(batch_size, height, width)
            C = torch.zeros(batch_size, 2, height, width)  # temporal order = 2
        else:
            H, C = init_states

        # Compute cell state
        C = C.permute(0, 3, 2, 1)
        Wcc = torch.transpose(self.Wcc, 0, 1)
        Wcc = Wcc.repeat(batch_size, width, 1, 1)
        C = torch.matmul(C, Wcc)
        C = C.permute(0, 3, 2, 1)
        wcu = self.wcu.repeat(batch_size, 1)
        wcu = wcu.unsqueeze(-1).unsqueeze(-1)
        U = inputs.unsqueeze(1)
        U = U.repeat(1, 2, 1, 1)  # index 1 is temporal order
        C = C + wcu * U

        # Estimate source map
        V = inputs - H

        # Compute homogeneous solution
        whc = self.whc.repeat(batch_size, width, 1)
        whc = whc.unsqueeze(-1)
        H = torch.matmul(C.permute(0, 3, 2, 1), whc)
        H = H.permute(3, 0, 2, 1)[0]
        H = H + self.c2 * self.spatial(inputs.unsqueeze(1))[:, 0]

        # Predict next source map
        V_hat = V / self.sf
        V_hat = self.RedNet(V_hat.unsqueeze(1))
        V_hat = V_hat[:,0]
        V_hat = V_hat * self.sf

        # Compute output
        outputs = H + V_hat

        return outputs, H, C, V, V_hat


