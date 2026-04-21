import torch.nn as nn
import torch

'''
Networks for the Burgers equation
'''
import numpy as np

import torch
import torch.nn as nn

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FfLayer(nn.Module):
    """
    PyTorch Module for Fourier feature extraction.
    """
    def __init__(self, input_dim=10, m=100, sig=None, B_matrices=None):
        super(FfLayer, self).__init__()
        self.input_dim = input_dim
        self.m = m
        self.sig = sig if sig is not None else [1, 10, 20, 50, 100]
        
        if B_matrices is None:
            B_matrices_list = []
            for sig_i in self.sig:
                B_matrices_list.append(torch.randn(self.input_dim, self.m) * sig_i)
            B_matrices = torch.cat(B_matrices_list, dim=1)
        
        self.register_buffer('B_matrices', B_matrices)

    def forward(self, inputs):
        inputs_x_B = torch.matmul(inputs, self.B_matrices)
        aux = torch.cat([torch.cos(inputs_x_B), torch.sin(inputs_x_B)], dim=0)
        inp_n_rows = inputs.size(0)
        return torch.transpose(aux.view(self.m * len(self.sig) * 2, inp_n_rows), 0, 1)

class KSNet(nn.Module):
    """
    PyTorch model for a physics-informed neural network (PINN)
    for the Kuramoto-Sivashinsky equation.
    """
    def __init__(self,layers=[50]*4, activation=torch.tanh,
                 sig_t=[1, 10], sig_x=[1, 20], num_outputs=1):
        super().__init__()
        if layers is None:
            layers = [40, 40, 40, 40]

        # Fourier feature layers for time and space
        self.t_fflayer = FfLayer(input_dim=1, m=layers[0], sig=sig_t)
        self.x_fflayer = FfLayer(input_dim=1, m=layers[0], sig=sig_x)

        # Dense neural network layers
        fnn_layers = []
        in_features = layers[0] * 2 * len(sig_t)
        for layer_size in layers:
            fnn_layers.append(nn.Linear(in_features, layer_size))
            fnn_layers.append(nn.Tanh())
            in_features = layer_size

        self.fnn = nn.Sequential(*fnn_layers)
        self.output_layer = nn.Linear(layers[-1], num_outputs)

    def forward(self, x, t):
        ini_shape=x.shape
        # Split inputs into time (t) and space (x)
        #t, x = torch.split(inputs, 1, dim=1)
        x=x.view(-1).unsqueeze(1)/2/np.pi
        t=t.view(-1).unsqueeze(1)

        # Pass through Fourier feature layers
        t = self.t_fflayer(t)
        x = self.x_fflayer(x)

        # Forward pass through FNN for time and space
        t = self.fnn(t)
        x = self.fnn(x)

        # Pointwise multiplication to merge
        tx = torch.mul(t, x)

        # Output layer
        outputs = self.output_layer(tx)
        return outputs.view(ini_shape)

'''
class KSNet(nn.Module):
    
    def __init__(self,channel_basics=100,n_layers=5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ini_net=nn.Sequential(nn.Linear(2, channel_basics), nn.Tanh())
        self.net=[]
        for i in range(n_layers):
            self.net.append(nn.Sequential(nn.Linear(channel_basics, channel_basics), nn.Tanh()))
        self.net=nn.Sequential(*self.net)
        self.out_net=nn.Linear(channel_basics, 1, bias=False)
    
    def forward(self, x, t):
        ini_shape=x.shape
        y = torch.stack([x.view(-1), t.view(-1)], dim=-1)        
        y = torch.stack([x, t], dim=-1)
        y = self.ini_net(y)
        y = self.net(y)
        y = self.out_net(y)
        return y.view(ini_shape)
'''
from collections import OrderedDict
import torch


class LAAFlayer(torch.nn.Module):

    def __init__(self, n, a, dim_in, dim_out, activation):
        super(LAAFlayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n = n
        self.register_parameter('a', torch.nn.Parameter(a))
        self.activation = activation

        self.fc = torch.nn.Linear(self.dim_in, self.dim_out)

    def forward(self, x):
        x1 = self.fc(x)
        x2 = self.n * torch.mul(self.a, x1)
        out = self.activation(x2)
        return out


# n layers deep neural network with LAAF
class KSNet(torch.nn.Module):

    def __init__(self, n_layers=5, n_hidden=100, x_dim=2, u_dim=1):
        super(KSNet, self).__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.activation = torch.nn.Tanh
        self.regularizer = None

        #self.a = torch.nn.Parameter(torch.empty(size=(self.n_layers, self.n_hidden)))
        self.a=torch.empty(size=(self.n_layers, self.n_hidden))
        #self.register_parameter('a', self.a)
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.4)

        layer_list = list()
        layer_list.append(('layer0', LAAFlayer(10, self.a[0, :], x_dim, n_hidden, self.activation())))
        for i in range(self.n_layers - 1):
            layer_list.append(('layer%d' % (i + 1), LAAFlayer(10, self.a[i + 1, :], n_hidden, n_hidden, self.activation())))
        layer_list.append(('layer%d' % n_layers, torch.nn.Linear(self.n_hidden, self.u_dim)))
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x,t):
        ini_shape=x.shape
        out = self.layers(torch.stack([x.view(-1), t.view(-1)], dim=-1))
        return out.view(ini_shape)
'''  
'''    
class AdaptiveLinear(nn.Linear):
    r"""Applies a linear transformation to the input data as follows
    :math:`y = naxA^T + b`.
    More details available in Jagtap, A. D. et al. Locally adaptive
    activation functions with slope recovery for deep and
    physics-informed neural networks, Proc. R. Soc. 2020.

    Parameters
    ----------
    in_features : int
        The size of each input sample
    out_features : int 
        The size of each output sample
    bias : bool, optional
        If set to ``False``, the layer will not learn an additive bias
    adaptive_rate : float, optional
        Scalable adaptive rate parameter for activation function that
        is added layer-wise for each neuron separately. It is treated
        as learnable parameter and will be optimized using a optimizer
        of choice
    adaptive_rate_scaler : float, optional
        Fixed, pre-defined, scaling factor for adaptive activation
        functions
    """
    def __init__(self, in_features, out_features, bias=True, adaptive_rate=None, adaptive_rate_scaler=None):
        super(AdaptiveLinear, self).__init__(in_features, out_features, bias)
        self.adaptive_rate = adaptive_rate
        self.adaptive_rate_scaler = adaptive_rate_scaler
        if self.adaptive_rate:
            self.A = nn.Parameter(self.adaptive_rate * torch.ones(self.in_features))
            if not self.adaptive_rate_scaler:
                self.adaptive_rate_scaler = 10.0
            
    def forward(self, input):
        if self.adaptive_rate:
            return nn.functional.linear(self.adaptive_rate_scaler * self.A * input, self.weight, self.bias)
        return nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, '
            f'adaptive_rate={self.adaptive_rate is not None}, adaptive_rate_scaler={self.adaptive_rate_scaler is not None}'
        )
        
class KSNet(nn.Module):
    
    def __init__(self,channel_basics=100,n_layers=5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ini_net=nn.Sequential(AdaptiveLinear(2, channel_basics,adaptive_rate=0.1,adaptive_rate_scaler=10.0), nn.Tanh())
        self.net=[]
        for i in range(n_layers):
            self.net.append(nn.Sequential(AdaptiveLinear(channel_basics, channel_basics,adaptive_rate=0.1,adaptive_rate_scaler=10.0), nn.Tanh()))
        self.net=nn.Sequential(*self.net)
        self.out_net=nn.Sequential(nn.Linear(channel_basics, 1, bias=False))
    
    def forward(self, x, t):
        ini_shape=x.shape
        x=2*((x/2/np.pi)-0.5)
        t=2*(t-0.5)
        y = torch.stack([x.view(-1), t.view(-1)], dim=-1)        
        y = torch.stack([x, t], dim=-1)
        y = self.ini_net(y)
        y = self.net(y)
        y = self.out_net(y)
        return y.view(ini_shape)
'''  