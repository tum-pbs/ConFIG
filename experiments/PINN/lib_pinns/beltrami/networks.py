import torch.nn as nn
import torch

'''
Networks for the Beltrami equation
'''

class BeltramiNet(nn.Module):
    
    def __init__(self,channel_basics=50,n_layers=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ini_net=nn.Sequential(nn.Linear(4, channel_basics), nn.Tanh())
        self.net=[]
        for i in range(n_layers):
            self.net.append(nn.Sequential(
                nn.Linear(channel_basics, channel_basics), 
                nn.Tanh()
                ))
        self.net=nn.Sequential(*self.net)
        self.out_net=nn.Sequential(nn.Linear(channel_basics, 4))
    
    def forward(self, x, y,z,t):
        ini_shape=x.shape   
        y = torch.stack([x, y,z,t], dim=-1)
        y = self.ini_net(y)
        y = self.net(y)
        y = self.out_net(y)
        return y