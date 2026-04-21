import torch.nn as nn
import torch
from .simulation_paras import *
'''
Networks for the Kovasznay equation
'''

class HeatMSNet(nn.Module):
    
    def __init__(self,channel_basics=50,n_layers=4,simulation_time=SIMULATION_TIME,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ini_net=nn.Sequential(nn.Linear(3, channel_basics), nn.Tanh())
        self.net=[]
        for i in range(n_layers):
            self.net.append(nn.Sequential(
                nn.Linear(channel_basics, channel_basics), 
                nn.Tanh()
                ))
        self.net=nn.Sequential(*self.net)
        self.out_net=nn.Linear(channel_basics, 1,bias=False)
        self.simulation_time=simulation_time
    
    def forward(self, x, t):
        t=t/self.simulation_time
        y = self.ini_net(torch.cat([x,t],dim=1))
        y = self.net(y)
        y = self.out_net(y)
        return y