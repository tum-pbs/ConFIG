#usr/bin/python3

#version:0.0.1
#last modified:20230803

from . import *
from einops import rearrange
from abc import abstractmethod


class TimeEncoding(nn.Module):
    def __init__(self, dim_encoding:int):
        super().__init__()
        self.dim = dim_encoding

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimeEmbedding(nn.Module):

    def __init__(self,dim_input:None,dim_encoded_time=None,trainable=True):
        super().__init__()
        if trainable:
            if dim_input is None or dim_encoded_time is None:
                raise RuntimeError("'dim_input' and 'dim_encoded_time' must be specficed when trainable is True")
            else:
                self.linear1=nn.Linear(dim_encoded_time,dim_input)
                self.activation=nn.GELU()
                self.linear2=nn.Linear(dim_input,dim_input)
        else:
            self.linear1=nn.Identity()
            self.activation=nn.Identity()
            self.linear2=nn.Identity()
    

    def forward(self,x, t):
        t=self.linear2(self.activation(self.linear1(t)))
        return x + rearrange(t, "b c -> b c 1 1")

class TimeEmbModel(nn.Module):

    @abstractmethod
    def forward(self, x, t):
        """
        ...
        """

class ConditionEmbModel(nn.Module):

    @abstractmethod
    def forward(self, x, condition):
        """
        ...
        """

class TimeConditionEmbModel(nn.Module):

    @abstractmethod
    def forward(self, x, t, condition):
        """
        ...
        """

class EmbedSequential(nn.Sequential,TimeConditionEmbModel):
    def forward(self, x, t, condition):
        for layer in self:
            if isinstance(layer, TimeEmbModel) or isinstance(layer, TimeEmbedding):
                x = layer(x, t)
            elif isinstance(layer, ConditionEmbModel):
                x = layer(x, condition)
            elif isinstance(layer, TimeConditionEmbModel):
                x = layer(x,t,condition)
            else:
                x = layer(x)
        return x