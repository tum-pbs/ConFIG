from .simulation_paras import *
from ..helpers import derivative
import torch

def physical_residual(u, x):
    x.grad = None
    return derivative(u, x, order=2).sum(dim=1)+torch.pi**2/4*torch.sin(torch.pi/2*x).sum(dim=1)
