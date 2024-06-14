import torch.nn as nn
import torch

'''
Networks for the Burgers equation
'''

class BurgersNetRes(nn.Module):
    def __init__(self, num_channels_input=2, channel_basics=20, channel_multiplier=[1]*4,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_layers=len(channel_multiplier)-1
        self.layer_in = nn.Sequential(
            nn.Linear(num_channels_input, channel_multiplier[0]*channel_basics),
            nn.GroupNorm(num_groups=4,num_channels=channel_multiplier[0]*channel_basics),
            nn.LeakyReLU(),
        )
        self.res_blocks=nn.ModuleList()
        for i in range(self.num_layers):
            in_channel=channel_multiplier[i]*channel_basics
            out_channel=channel_multiplier[i+1]*channel_basics
            self.res_blocks.append(
                nn.Sequential(
                    nn.Linear(in_channel, in_channel),
                    nn.GroupNorm(num_groups=4,num_channels=in_channel),
                    nn.LeakyReLU(),
                    nn.Linear(in_channel, out_channel),
                    nn.GroupNorm(num_groups=4,num_channels=out_channel),
                )
                )
        self.layer_out = nn.Linear(channel_multiplier[-1]*channel_basics, 1)
            
    def forward(self, x, t):
        ini_shape=x.shape
        y = torch.stack([x.view(-1), t.view(-1)], dim=-1)
        y= self.layer_in(y)
        for res_block in self.res_blocks:
            y=res_block(y)
            y=nn.functional.leaky_relu(y)
        y = self.layer_out(y)
        return y.view(ini_shape)

class BurgersNet(nn.Module):
    
    def __init__(self,channel_basics=50,n_layers=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ini_net=nn.Sequential(nn.Linear(2, channel_basics), nn.Tanh())
        self.net=[]
        for i in range(n_layers):
            self.net.append(nn.Sequential(nn.Linear(channel_basics, channel_basics), nn.Tanh()))
        self.net=nn.Sequential(*self.net)
        self.out_net=nn.Sequential(nn.Linear(channel_basics, 1))
    
    def forward(self, x, t):
        ini_shape=x.shape
        y = torch.stack([x.view(-1), t.view(-1)], dim=-1)        
        y = torch.stack([x, t], dim=-1)
        y = self.ini_net(y)
        y = self.net(y)
        y = self.out_net(y)
        return y.view(ini_shape)
