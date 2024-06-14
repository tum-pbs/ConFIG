#usr/bin/python3
# -*- coding: UTF-8 -*-
from . import *
    
def get_para_vector(network):
    """
    Returns the parameter vector of the given network.

    Args:
        network (torch.nn.Module): The network for which to compute the gradient vector.

    Returns:
        torch.Tensor: The parameter vector of the network.
    """
    with torch.no_grad():
        para_vec = None
        for par in network.parameters():
            viewed=par.data.view(-1)
            if para_vec is None:
                para_vec = viewed
            else:
                para_vec = torch.cat((para_vec, viewed))
        return para_vec
    
def get_gradient_vector(network):
    """
    Returns the gradient vector of the given network.

    Args:
        network (torch.nn.Module): The network for which to compute the gradient vector.

    Returns:
        torch.Tensor: The gradient vector of the network.
    """
    with torch.no_grad():
        grad_vec = None
        for par in network.parameters():
            viewed=par.grad.data.view(-1)
            if grad_vec is None:
                grad_vec = viewed
            else:
                grad_vec = torch.cat((grad_vec, viewed))
        return grad_vec

def apply_gradient_vector(network:torch.nn.Module,grad_vec:torch.Tensor):
    """
    Applies a gradient vector to the network's parameters.

    Args:
        network (torch.nn.Module): The network to apply the gradient vector to.
        grad_vec (torch.Tensor): The gradient vector to apply.

    """
    with torch.no_grad():
        start=0
        for par in network.parameters():
            end=start+par.grad.data.view(-1).shape[0]
            par.grad.data=grad_vec[start:end].view(par.grad.data.shape)
            start=end

def get_cos_angle(vector1:torch.Tensor,vector2:torch.Tensor):
    """
    Calculates the cosine angle between two vectors.

    Args:
        vector1 (torch.Tensor): The first vector.
        vector2 (torch.Tensor): The second vector.

    Returns:
        torch.Tensor: The cosine angle between the two vectors.
    """
    with torch.no_grad():
        return torch.dot(vector1,vector2)/vector1.norm()/vector2.norm()
    
def unit_vector(vector: torch.Tensor):
    """
    Compute the unit vector of a given tensor.

    Parameters:
        vector (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The unit vector of the input tensor.
    """
    with torch.no_grad():
        if vector.norm()==0:
            print("Warning: detected zero vector.")
            return torch.zeros_like(vector)
        else:
            return vector / vector.norm()

def transfer_coef_double(weights: torch.tensor, 
                         unit_vec_1: torch.tensor, unit_vec_2: torch.tensor,
                         or_unit_vec_1: torch.tensor, or_unit_vec_2: torch.tensor):
    """
    Transfer the angle weights to a length coefficient for ConFIG method with two vectors.
    
    This function will return a coefficient matrix [c_1,c_2] so that
    $$
    \frac{(c_1 o_1+c_2 o_2)\dot \mathbf{v}_1}{(c_1 o_1+c_2 o_2) \dot \mathbf{v}_2}
    =\frac{w_1}{w_2}
    $$
    where w_1 and w_2 are the angle weights,
    v_1 and v_2 are unit vectors of the two gradients,
    and o_1 and o_2 are the unit orthogonal components of the two gradients.

    Args:
        weights (torch.tensor): Angle weights for the two vectors. It should have shape (2,).
        unit_vecs (torch.tensor): Unit vectors of the orthogonal components. It should have shape (2,k) where k is the length of the vector.

    Returns:
        torch.tensor: The coefficients for the two vectors.

    Raises:
        ValueError: If the number of weights or unit vectors is not equal to 2.
    """    

    return torch.dot(or_unit_vec_2,unit_vec_1)/(weights[0]/weights[1]*torch.dot(or_unit_vec_1,unit_vec_2)),1

def estimate_conflict(gradients: torch.Tensor):
    """
    Estimates the degree of conflict of gradients.

    Args:
        gradients (torch.Tensor): A tensor containing gradients.

    Returns:
        torch.Tensor: A tensor consistent of the dot products between the sum of gradients and each sub-gradient.
    """
    direct_sum = unit_vector(gradients.sum(dim=0))
    unit_grads = gradients / torch.norm(gradients, dim=1).view(-1, 1)
    return unit_grads @ direct_sum


def has_zero(lists:Sequence):
    """
    Check if any element in the list is zero.

    Args:
        lists (Sequence): A list of elements.

    Returns:
        bool: True if any element is zero, False otherwise.
    """
    for i in lists:
        if i==0:
            return True
    return False

class OrderedSliceSelector:
    def __init__(self):
        self.start_index=0
        
    def select(self, n:int, source_sequence:Sequence):
        if n > len(source_sequence):
            raise ValueError("n must be less than or equal to the length of the source sequence")
        end_index = self.start_index + n
        if end_index > len(source_sequence)-1:
            new_start=end_index-len(source_sequence)
            indexes = list(range(self.start_index,len(source_sequence)))+list(range(0,new_start))
            self.start_index=new_start
        else:
            indexes = list(range(self.start_index,end_index))
            self.start_index=end_index
        return indexes,[source_sequence[i] for i in indexes]