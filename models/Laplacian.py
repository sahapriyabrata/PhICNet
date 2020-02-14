import torch
import torch.nn as nn

class Laplacian(nn.Module):

    def __init__(self):
        super(Laplacian, self).__init__()
        self.kernel = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.laplacian = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.laplacian.weight = nn.Parameter(self.kernel.unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, maps):
        return self.laplacian(maps)

