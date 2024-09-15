#usr/bin/python3
# -*- coding: UTF-8 -*-
from . import *

from typing import Union
from .utils import *
from .weight_model import *
from .length_model import *

def ConFIG_update_double(grad_1:torch.Tensor,grad_2:torch.Tensor,
                         weight_model:WeightModel=EqualWeight(),
                         length_model:LengthModel=ProjectionLength(),
                         losses:Optional[Sequence]=None)-> torch.Tensor:
    """
    ConFIG update for two gradients where no inverse calculation is needed. 

    Args:
        grad_1 (torch.Tensor): The first gradient.
        grad_2 (torch.Tensor): The second gradient.
        weight_model (WeightModel, optional): The weight model for calculating the direction weights.
            Defaults to EqualWeight(), which will make the final update gradient not biased towards any gradient.
        length_model (LengthModel, optional): The length model for rescaling the length of the final gradient. 
            Defaults to ProjectionLength(), which will project each gradient vector onto the final gradient vector to get the final length.
        losses (Optional[Sequence], optional): The losses associated with the gradients. 
            The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
            you can set this value as None. Defaults to None.

    Returns:
        torch.Tensor: The final update gradient.
        
    Examples:
        ```python
        from conflictfree.grad_operator import ConFIG_update_double
        from conflictfree.utils import get_gradient_vector,apply_gradient_vector
        optimizer=torch.Adam(network.parameters(),lr=1e-3)
        for input_i in dataset:
            grads=[] # we record gradients rather than losses
            for loss_fn in [loss_fn1, loss_fn2]:
                optimizer.zero_grad()
                loss_i=loss_fn(input_i)
                loss_i.backward()
                grads.append(get_gradient_vector(network)) #get loss-specfic gradient
            g_config=ConFIG_update_double(grads) # calculate the conflict-free direction
            apply_gradient_vector(network) # set the condlict-free direction to the network
            optimizer.step()
        ```

    """
    with torch.no_grad():
        norm_1=grad_1.norm();norm_2=grad_2.norm()
        unit_1=grad_1/norm_1;unit_2=grad_2/norm_2
        cos_angle=get_cos_similarity(grad_1,grad_2)
        or_2=grad_1-norm_1*cos_angle*unit_2
        or_1=grad_2-norm_2*cos_angle*unit_1
        unit_or1=unit_vector(or_1);unit_or2=unit_vector(or_2)
        coef_1,coef_2=transfer_coef_double(
            weight_model.get_weights(gradients=torch.stack([grad_1,grad_2]),losses=losses,device=grad_1.device),
            unit_1,
            unit_2,
            unit_or1,
            unit_or2,)
        best_direction=coef_1*unit_or1+coef_2*unit_or2
        return length_model.rescale_length(target_vector=best_direction,
                                           gradients=torch.stack([grad_1,grad_2]),
                                           losses=losses,
                                           )

def ConFIG_update(
    grads:Union[torch.Tensor,Sequence[torch.Tensor]],
    weight_model:WeightModel=EqualWeight(),
    length_model:LengthModel=ProjectionLength(),
    use_least_square:bool=True,
    losses:Optional[Sequence]=None)-> torch.Tensor:
    """
    Performs the standard ConFIG update step.

    Args:
        grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update. 
            It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
        weight_model (WeightModel, optional): The weight model for calculating the direction weights.
            Defaults to EqualWeight(), which will make the final update gradient not biased towards any gradient.
        length_model (LengthModel, optional): The length model for rescaling the length of the final gradient. 
            Defaults to ProjectionLength(), which will project each gradient vector onto the final gradient vector to get the final length.
        use_least_square (bool, optional): Whether to use the least square method for calculating the best direction. 
            If set to False, we will directly calculate the pseudo-inverse of the gradient matrix. See `torch.linalg.pinv` and `torch.linalg.lstsq` for more details.
            Recommended to set to True. Defaults to True.
        losses (Optional[Sequence], optional): The losses associated with the gradients. 
            The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
            you can set this value as None. Defaults to None.

    Returns:
        torch.Tensor: The final update gradient.
        
    Examples:
        ```python
        from conflictfree.grad_operator import ConFIG_update
        from conflictfree.utils import get_gradient_vector,apply_gradient_vector
        optimizer=torch.Adam(network.parameters(),lr=1e-3)
        for input_i in dataset:
            grads=[] # we record gradients rather than losses
            for loss_fn in loss_fns:
                optimizer.zero_grad()
                loss_i=loss_fn(input_i)
                loss_i.backward()
                grads.append(get_gradient_vector(network)) #get loss-specfic gradient
            g_config=ConFIG_update(grads) # calculate the conflict-free direction
            apply_gradient_vector(network) # set the condlict-free direction to the network
            optimizer.step()
        ```
    """
    if not isinstance(grads,torch.Tensor):
        grads=torch.stack(grads)
    with torch.no_grad():
        weights=weight_model.get_weights(gradients=grads,losses=losses,device=grads.device)
        units=torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)),0)
        if use_least_square:
            best_direction=torch.linalg.lstsq(units, weights).solution
        else:
            best_direction=torch.linalg.pinv(units)@weights
        return length_model.rescale_length(target_vector=best_direction,
                                           gradients=grads,
                                           losses=losses,
                                            )

class GradientOperator:
    """
    A base class that represents a gradient operator.

    Methods:
        calculate_gradient: Calculates the gradient based on the given gradients and losses.
        update_gradient: Updates the gradient of the network based on the calculated gradient.

    """

    def __init__(self):
        pass
    
    def calculate_gradient(self, grads: Union[torch.Tensor,Sequence[torch.Tensor]], losses: Optional[Sequence] = None)-> torch.Tensor:
        """
        Calculates the gradient based on the given gradients and losses.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update. 
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients. 
                The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
                you can set this value as None. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("calculate_gradient method must be implemented")
    
    def update_gradient(self, network: torch.nn.Module, grads: Union[torch.Tensor,Sequence[torch.Tensor]], losses: Optional[Sequence] = None)-> None:
        """
        Calculate the gradient and apply the gradient to the network.

        Args:
            network (torch.nn.Module): The target network.
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update. 
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients. 
                The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
                you can set this value as None. Defaults to None.

        Returns:
            None

        """
        apply_gradient_vector(network, self.calculate_gradient(grads, losses))
    

class ConFIGOperator(GradientOperator):
    """
    Operator for the ConFIG algorithm.

    Args:
        weight_model (WeightModel, optional): The weight model for calculating the direction weights.
            Defaults to EqualWeight(), which will make the final update gradient not biased towards any gradient.
        length_model (LengthModel, optional): The length model for rescaling the length of the final gradient. 
            Defaults to ProjectionLength(), which will project each gradient vector onto the final gradient vector to get the final length.
        allow_simplified_model (bool, optional): Whether to allow simplified model for calculating the gradient. 
            If set to True, will use simplified form of ConFIG method when there are only two losses (ConFIG_update_double). Defaults to True.
        use_least_square (bool, optional): Whether to use the least square method for calculating the best direction. 
            If set to False, we will directly calculate the pseudo-inverse of the gradient matrix. See `torch.linalg.pinv` and `torch.linalg.lstsq` for more details.
            Recommended to set to True. Defaults to True.
    
    Examples:
        ```python
        from conflictfree.grad_operator import ConFIGOperator
        from conflictfree.utils import get_gradient_vector,apply_gradient_vector
        optimizer=torch.Adam(network.parameters(),lr=1e-3)
        operator=ConFIGOperator() # initialize operator
        for input_i in dataset:
            grads=[]
            for loss_fn in loss_fns:
                optimizer.zero_grad()
                loss_i=loss_fn(input_i)
                loss_i.backward()
                grads.append(get_gradient_vector(network))
            g_config=operator.calculate_gradient(grads) # calculate the conflict-free direction
            apply_gradient_vector(network) # or simply use `operator.update_gradient(network,grads)` to calculate and set the condlict-free direction to the network
            optimizer.step()
        ```
    
    """

    def __init__(self, 
                 weight_model: WeightModel = EqualWeight(),
                 length_model: LengthModel = ProjectionLength(),
                 allow_simplified_model: bool = True,
                 use_least_square: bool = True):
        super().__init__()
        self.weight_model = weight_model
        self.length_model = length_model
        self.allow_simplified_model = allow_simplified_model
        self.use_least_square = use_least_square

    def calculate_gradient(self, grads: Union[torch.Tensor,Sequence[torch.Tensor]], losses: Optional[Sequence] = None)->torch.Tensor:
        """
        Calculates the gradient using the ConFIG algorithm.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update. 
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients. 
                The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
                you can set this value as None. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient.
        """
        if not isinstance(grads,torch.Tensor):
            grads=torch.stack(grads)
        if grads.shape[0] == 2 and self.allow_simplified_model:
            return ConFIG_update_double(grads[0], grads[1],
                                        weight_model=self.weight_model,
                                        length_model=self.length_model,
                                        losses=losses)
        else:
            return ConFIG_update(grads,
                                 weight_model=self.weight_model,
                                 length_model=self.length_model,
                                 use_least_square=self.use_least_square,
                                 losses=losses)

class PCGradOperator(GradientOperator):
    """
    PCGradOperator class represents a gradient operator for PCGrad algorithm.
    
    @inproceedings{yu2020gradient,
    title={Gradient surgery for multi-task learning},
    author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
    booktitle={34th International Conference on Neural Information Processing Systems},
    year={2020},
    url={https://dl.acm.org/doi/abs/10.5555/3495724.3496213}
    }

    """


    def calculate_gradient(self, grads: Union[torch.Tensor,Sequence[torch.Tensor]], losses: Optional[Sequence] = None)->torch.Tensor:
        """
        Calculates the gradient using the PCGrad algorithm.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update. 
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): This parameter should not be set for current operator. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient using PCGrad method.
        """
        if not isinstance(grads,torch.Tensor):
            grads=torch.stack(grads)
        with torch.no_grad():
            grads_pc=torch.clone(grads)
            length=grads.shape[0]
            for i in range(length):
                for j in range(length):
                    if j !=i:
                        dot=grads_pc[i].dot(grads[j])
                        if dot<0:
                            grads_pc[i]-=dot*grads[j]/((grads[j].norm())**2)
            return torch.sum(grads_pc,dim=0)
        
class IMTLGOperator(GradientOperator):
    """
    PCGradOperator class represents a gradient operator for IMTL-G algorithm.
    
    @inproceedings{
    liu2021towards,
    title={Towards Impartial Multi-task Learning},
    author={Liyang Liu and Yi Li and Zhanghui Kuang and Jing-Hao Xue and Yimin Chen and Wenming Yang and Qingmin Liao and Wayne Zhang},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=IMPnRXEWpvr}
    }

    """


    def calculate_gradient(self, grads: Union[torch.Tensor,Sequence[torch.Tensor]], losses: Optional[Sequence] = None) ->torch.Tensor:
        """
        Calculates the gradient using the IMTL-G algorithm.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update. 
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): This parameter should not be set for current operator. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient using IMTL-G method.
        """
        if not isinstance(grads,torch.Tensor):
            grads=torch.stack(grads)
        with torch.no_grad():
            ut_norm=grads/grads.norm(dim=1).unsqueeze(1)
            ut_norm=torch.nan_to_num(ut_norm,0)
            ut=torch.stack([ut_norm[0]-ut_norm[i+1] for i in range(grads.shape[0]-1)],dim=0).T
            d=torch.stack([grads[0]-grads[i+1] for i in range(grads.shape[0]-1)],dim=0)
            at=grads[0]@ut@torch.linalg.pinv(d@ut)
            return (1-torch.sum(at))*grads[0]+torch.sum(at.unsqueeze(1)*grads[1:],dim=0)