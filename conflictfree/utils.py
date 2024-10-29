# usr/bin/python3
# -*- coding: UTF-8 -*-
from . import *
import numpy as np
from typing import Literal


def get_para_vector(network: torch.nn.Module) -> torch.Tensor:
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
            viewed = par.data.view(-1)
            if para_vec is None:
                para_vec = viewed
            else:
                para_vec = torch.cat((para_vec, viewed))
        return para_vec


def get_gradient_vector(
    network: torch.nn.Module, none_grad_mode: Literal["raise", "zero", "skip"] = "skip"
) -> torch.Tensor:
    """
    Returns the gradient vector of the given network.

    Args:
        network (torch.nn.Module): The network for which to compute the gradient vector.
        none_grad_mode (Literal['raise', 'zero', 'skip']): The mode to handle None gradients. default: 'skip'
            - 'raise': Raise an error when the gradient of a parameter is None.
            - 'zero': Replace the None gradient with a zero tensor.
            - 'skip': Skip the None gradient.
                        The None gradient usually occurs when part of the network is not trainable (e.g., fine-tuning)
            or the weight is not used to calculate the current loss (e.g., different parts of the network calculate different losses).
            If all of your losses are calculated using the same part of the network, you should set none_grad_mode to 'skip'.
            If your losses are calculated using different parts of the network, you should set none_grad_mode to 'zero' to ensure the gradients have the same shape.

    Returns:
        torch.Tensor: The gradient vector of the network.
    """
    with torch.no_grad():
        grad_vec = None
        for par in network.parameters():
            if par.grad is None:
                if none_grad_mode == "raise":
                    raise RuntimeError("None gradient detected.")
                elif none_grad_mode == "zero":
                    viewed = torch.zeros_like(par.data.view(-1))
                elif none_grad_mode == "skip":
                    continue
                else:
                    raise ValueError(f"Invalid none_grad_mode '{none_grad_mode}'.")
            else:
                viewed = par.grad.data.view(-1)
            if grad_vec is None:
                grad_vec = viewed
            else:
                grad_vec = torch.cat((grad_vec, viewed))
        return grad_vec


def apply_gradient_vector(
    network: torch.nn.Module,
    grad_vec: torch.Tensor,
    none_grad_mode: Literal["zero", "skip"] = "skip",
    zero_grad_mode: Literal["skip", "pad_zero", "pad_value"] = "pad_value",
) -> None:
    """
    Applies a gradient vector to the network's parameters.
    This function requires the network contains the some gradient information in order to apply the gradient vector.
    If your network does not contain the gradient information, you should consider using `apply_gradient_vector_para_based` function.

    Args:
        network (torch.nn.Module): The network to apply the gradient vector to.
        grad_vec (torch.Tensor): The gradient vector to apply.
        none_grad_mode (Literal['zero', 'skip']): The mode to handle None gradients.
            You should set this parameter to the same value as the one used in `get_gradient_vector` method.
        zero_grad_mode (Literal['padding', 'skip']): How to set the value of the gradient if your `none_grad_mode` is "zero". default: 'skip'
            - 'skip': Skip the None gradient.
            - 'padding': Replace the None gradient with a zero tensor.
            - 'pad_value': Replace the None gradient using the value in the gradient.
            If you set `none_grad_mode` to 'zero', that means you padded zero to your `grad_vec` if the gradient of the parameter is None when getting the gradient vector.
            When you apply the gradient vector back to the network, the value in the `grad_vec` corresponding to the previous None gradient may not be zero due to the applied gradient operation.
                        Thus, you need to determine whether to recover the original None value, set it to zero, or set the value according to the value in `grad_vec`.
            If you are not sure what you are doing, it is safer to set it to 'pad_value'.

    """
    if none_grad_mode == "zero" and zero_grad_mode == "pad_value":
        apply_gradient_vector_para_based(network, grad_vec)
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            if par.grad is None:
                if none_grad_mode == "skip":
                    continue
                elif none_grad_mode == "zero":
                    start = start + par.data.view(-1).shape[0]
                    if zero_grad_mode == "pad_zero":
                        par.grad = torch.zeros_like(par.data)
                    elif zero_grad_mode == "skip":
                        continue
                    else:
                        raise ValueError(f"Invalid zero_grad_mode '{zero_grad_mode}'.")
                else:
                    raise ValueError(f"Invalid none_grad_mode '{none_grad_mode}'.")
            else:
                end = start + par.data.view(-1).shape[0]
                par.grad.data = grad_vec[start:end].view(par.data.shape)
                start = end


def apply_gradient_vector_para_based(
    network: torch.nn.Module,
    grad_vec: torch.Tensor,
) -> None:
    """
    Applies a gradient vector to the network's parameters.
    Please only use this function when you are sure that the length of `grad_vec` is the same of your network's parameters.
    This happens when you use `get_gradient_vector` with `none_grad_mode` set to 'zero'.
    Or, the 'none_grad_mode' is 'skip' but all of the parameters in your network is involved in the loss calculation.

    Args:
        network (torch.nn.Module): The network to apply the gradient vector to.
        grad_vec (torch.Tensor): The gradient vector to apply.
    """
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            end = start + par.data.view(-1).shape[0]
            par.grad = grad_vec[start:end].view(par.data.shape)
            start = end


def apply_para_vector(network: torch.nn.Module, para_vec: torch.Tensor) -> None:
    """
    Applies a parameter vector to the network's parameters.

    Args:
        network (torch.nn.Module): The network to apply the parameter vector to.
        para_vec (torch.Tensor): The parameter vector to apply.
    """
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            end = start + par.data.view(-1).shape[0]
            par.data = para_vec[start:end].view(par.data.shape)
            start = end


def get_cos_similarity(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the cosine angle between two vectors.

    Args:
        vector1 (torch.Tensor): The first vector.
        vector2 (torch.Tensor): The second vector.

    Returns:
        torch.Tensor: The cosine angle between the two vectors.
    """
    with torch.no_grad():
        return torch.dot(vector1, vector2) / vector1.norm() / vector2.norm()


def unit_vector(vector: torch.Tensor, warn_zero: bool = False) -> torch.Tensor:
    """
    Compute the unit vector of a given tensor.

    Parameters:
        vector (torch.Tensor): The input tensor.
        warn_zero (bool): Whether to print a warning when the input tensor is zero. default: False

    Returns:
        torch.Tensor: The unit vector of the input tensor.
    """
    with torch.no_grad():
        if vector.norm() == 0:
            if warn_zero:
                print("Detected zero vector when doing normalization.")
            return torch.zeros_like(vector)
        else:
            return vector / vector.norm()


def transfer_coef_double(
    weights: torch.tensor,
    unit_vec_1: torch.tensor,
    unit_vec_2: torch.tensor,
    or_unit_vec_1: torch.tensor,
    or_unit_vec_2: torch.tensor,
) -> tuple:
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
        tuple: The coefficients for the two vectors.

    Raises:
        ValueError: If the number of weights or unit vectors is not equal to 2.
    """

    return (
        torch.dot(or_unit_vec_2, unit_vec_1)
        / (weights[0] / weights[1] * torch.dot(or_unit_vec_1, unit_vec_2)),
        1,
    )


def estimate_conflict(gradients: torch.Tensor) -> torch.Tensor:
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


def has_zero(lists: Sequence) -> bool:
    """
    Check if any element in the list is zero.

    Args:
        lists (Sequence): A list of elements.

    Returns:
        bool: True if any element is zero, False otherwise.
    """
    for i in lists:
        if i == 0:
            return True
    return False


class OrderedSliceSelector:
    """
    Selects a slice of the source sequence in order.
    Usually used for selecting loss functions/gradients/losses in momentum-based method if you want to update more tha one gradient in a single iteration.

    """

    def __init__(self):
        self.start_index = 0

    def select(
        self, n: int, source_sequence: Sequence
    ) -> Tuple[Sequence, Union[float, Sequence]]:
        """
        Selects a slice of the source sequence in order.

        Args:
            n (int): The length of the target slice.
            source_sequence (Sequence): The source sequence to select from.

        Returns:
            Tuple[Sequence,Union[float,Sequence]]: A tuple containing the indexes of the selected slice and the selected slice.
        """
        if n > len(source_sequence):
            raise ValueError(
                "n must be less than or equal to the length of the source sequence"
            )
        end_index = self.start_index + n
        if end_index > len(source_sequence) - 1:
            new_start = end_index - len(source_sequence)
            indexes = list(range(self.start_index, len(source_sequence))) + list(
                range(0, new_start)
            )
            self.start_index = new_start
        else:
            indexes = list(range(self.start_index, end_index))
            self.start_index = end_index
        if len(indexes) == 1:
            return indexes, source_sequence[indexes[0]]
        else:
            return indexes, [source_sequence[i] for i in indexes]


class RandomSliceSelector:
    """
    Selects a slice of the source sequence randomly.
    Usually used for selecting loss functions/gradients/losses in momentum-based method if you want to update more tha one gradient in a single iteration.
    """

    def select(
        self, n: int, source_sequence: Sequence
    ) -> Tuple[Sequence, Union[float, Sequence]]:
        """
        Selects a slice of the source sequence randomly.

        Args:
            n (int): The length of the target slice.
            source_sequence (Sequence): The source sequence to select from.

        Returns:
            Tuple[Sequence,Union[float,Sequence]]: A tuple containing the indexes of the selected slice and the selected slice.
        """
        assert n <= len(
            source_sequence
        ), "n can not be larger than or equal to the length of the source sequence"
        indexes = np.random.choice(len(source_sequence), n, replace=False)
        if len(indexes) == 1:
            return indexes, source_sequence[indexes[0]]
        else:
            return indexes, [source_sequence[i] for i in indexes]
