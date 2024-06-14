#usr/bin/python3

#version:0.0.1
#last modified:20230803

from . import *
from einops.layers.torch import Rearrange

def SPD_Conv_down_sampling(dim):
    """
    No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects
    https://arxiv.org/abs/2208.03641
    """
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim, 1),
    )

def interpolate_up_sampling(dim):
    
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim, 3, padding=1)  
    )