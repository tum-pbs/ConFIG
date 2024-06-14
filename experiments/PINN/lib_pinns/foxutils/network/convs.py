#usr/bin/python3

#version:0.0.1
#last modified:20230803

from . import *
from einops import reduce
from functools import partial
from .embedding import *
from .normal import *

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return torch.nn.functional.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class ResBlock(nn.Module):
    """
    full pre-activation Res
    https://arxiv.org/pdf/1603.05027.pdf
    """
    def __init__(self, dim_in, dim_out, use_WeightStandardizedConv=True):
        super().__init__()
        if use_WeightStandardizedConv:
            self.main_net=nn.Sequential(
                WeightStandardizedConv2d(dim_in, dim_out, 3, padding=1),
                GroupNormalX(dim_out),
                nn.GELU(),
                WeightStandardizedConv2d(dim_out, dim_out, 3, padding=1),
                GroupNormalX(dim_out),
            )
        else:
            self.main_net=nn.Sequential(
                WeightStandardizedConv2d(dim_in, dim_out, 3, padding=1),
                GroupNormalX(dim_out),
                nn.GELU(),
                WeightStandardizedConv2d(dim_out, dim_out, 3, padding=1),
                GroupNormalX(dim_out),
            )
        
        self.residual_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.act=nn.GELU()
    def forward(self, x):
        return self.act(self.main_net(x)+self.residual_conv(x))

class DSC_block(TimeEmbModel):
    #Depthwise Separable Convolution

    def __init__(self, dim_in, dim_out, dim_encoded_time):
        super().__init__()

        self.mainblock = self.build_main(dim_in=dim_in, dim_out=dim_out)
        self.timeEmb = TimeEmbedding(
            dim_input=dim_in, dim_encoded_time=dim_encoded_time, trainable=True)


    def build_main(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 7, padding=3, groups=dim_in),
            nn.GELU(),
            GroupNormalX(dim_in),
            nn.Conv2d(dim_in,dim_out,kernel_size=3,stride=1,padding=1),
        )

    def forward(self, x, t):
        return self.mainblock(self.timeEmb(x, t))
    
class DSC_Res_block(TimeEmbModel):
    #Depthwise Separable Convolution

    def __init__(self, dim_in, dim_out, dim_encoded_time):
        super().__init__()

        self.mainblock = self.build_main(dim_in=dim_in, dim_out=dim_out)
        self.residual_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.timeEmb = TimeEmbedding(
            dim_input=dim_in, dim_encoded_time=dim_encoded_time, trainable=True)


    def build_main(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 7, padding=3, groups=dim_in),
            nn.GELU(),
            GroupNormalX(dim_in),
            nn.Conv2d(dim_in,dim_out,kernel_size=3,stride=1,padding=1),
        )

    def forward(self, x, t):
        return self.mainblock(self.timeEmb(x, t))+self.residual_conv(x)
