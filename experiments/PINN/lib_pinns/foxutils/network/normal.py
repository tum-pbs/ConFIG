#usr/bin/python3

#version:0.0.1
#last modified:20230803

from . import *

class GroupNormalX(nn.Module):
    def __init__(self, dim, dim_eachgroup=16):
        super().__init__()
        groups = dim//dim_eachgroup
        if groups == 0:
            groups += 1
        self.norm = nn.GroupNorm(groups, dim)

    def forward(self, x):
        return self.norm(x)
