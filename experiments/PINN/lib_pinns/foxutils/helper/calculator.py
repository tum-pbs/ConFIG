#usr/bin/python3

#version:0.0.2
#last modified:20231210

import torch
def get_relative_error(a,b):
    """
    Calculate the mean relative error between two tensors.

    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.

    Returns:
        float: The mean relative error.
        torch.Tensor: The tensor containing the relative error for each element.
    """
    with torch.no_grad():
        ori_shape=a.shape
        a=a.reshape(-1)
        b=b.reshape(-1)
        relative_error=torch.zeros(1,device=a.device)
        relative_error_tensor=torch.zeros(b.shape,device=a.device)
        e_0=0
        for i,ai in enumerate(a):
            if ai.item()!=0:
                bi=b[i]
                error=torch.abs((bi-ai)/ai)
                relative_error += error
                relative_error_tensor[i]=error
            else:
                e_0+=1
        mean_relative_error=relative_error/(a.shape[0]-e_0)
        return mean_relative_error.item(),relative_error_tensor.reshape(ori_shape)

def redscale01(ori_tenser):
    '''
    Redscale a tensor to [0,1].

    Args:
        ori_tenser (torch.Tensor): The tensor to be redscaled.
    
    Returns:
        torch.Tensor: The redscaled tensor.
    '''
    return (ori_tenser-torch.min(ori_tenser))/(torch.max(ori_tenser)-torch.min(ori_tenser))
