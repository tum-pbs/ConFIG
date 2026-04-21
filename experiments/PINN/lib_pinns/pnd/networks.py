import torch.nn as nn
import torch
from .simulation_paras import *
'''
Networks for the Kovasznay equation
'''

class PNdNet(nn.Module):
    
    def __init__(self,channel_basics=50,n_layers=4, n_dim=N_DIM,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ini_net=nn.Sequential(nn.Linear(n_dim, channel_basics), nn.Tanh())
        self.net=[]
        for i in range(n_layers):
            self.net.append(nn.Sequential(
                nn.Linear(channel_basics, channel_basics), 
                nn.Tanh()
                ))
        self.net=nn.Sequential(*self.net)
        self.out_net=nn.Linear(channel_basics, 1,bias=False)
    
    def forward(self, x):
        x = self.ini_net(x)
        x = self.net(x)
        x = self.out_net(x)
        return x