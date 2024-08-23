from torch import Tensor
from torch.nn.modules import Module
from conflictfree import Sequence
from conflictfree.grad_operator import ConFIGOperator, GradientOperator
from conflictfree.utils import Sequence
from conflictfree.loss_recorder import LatestLossRecorder, LossRecorder
from . import *
from .utils import *
from .loss_recorder import *
from .grad_operator import *

class MomentumOperator():
    """
    The base class for the momentum based operator.
    
    Args:
        num_vectors (int): The number of gradient vectors.
        beta_1 (float): The moving average constant for the first momentum.
        beta_2 (float): The moving average constant for the second momentum.
        gradient_operator (GradientOperator, optional): The base gradient operator. Defaults to ConFIGOperator().
        loss_recorder (LossRecorder, optional): The loss recorder object. 
            If you want to pass loss information to "update_gradient" method or "apply_gradient" method, you need to specify a loss recorder. Defaults to None.

    Methods:
        calculate_gradient(indexes, grads, losses=None):
            Calculates the gradient based on the given indexes, gradients, and losses.
        update_gradient(network, indexes, grads, losses=None):
            Updates the gradient of the given network with the calculated gradient.
    """

    def __init__(self, 
                 num_vectors: int, 
                 beta_1: float=0.9, 
                 beta_2: float=0.999,
                 gradient_operator: GradientOperator=ConFIGOperator(),
                 loss_recorder: Optional[LossRecorder] = None) -> None:
        self.len_vectors = None
        self.device = None
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.num_vectors = num_vectors
        self.gradient_operator = gradient_operator
        self.loss_recorder = loss_recorder

    def calculate_gradient(self, 
                           indexes: Union[int,Sequence[int]], 
                           grads: Union[torch.Tensor,Sequence[torch.Tensor]], 
                           losses: Optional[Union[float,Sequence]] = None) -> torch.Tensor:
        """
        Calculates the gradient based on the given indexes, gradients, and losses.

        Args:
            indexes (Union[int,Sequence[int]]): The indexes of the gradient vectors and losses to be updated.
                The momentum with the given indexes will be updated based on the given gradients.
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update. 
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients. 
                The losses will be passed to base gradient operator. If the base gradient operator doesn't require loss information,
                you can set this value as None. Defaults to None.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
            
        Returns:
            torch.Tensor: The calculated gradient.
        """

        raise NotImplementedError("calculate_gradient method must be implemented")

    def _preprocess_gradients_losses(self, 
                                     indexes: Union[int,Sequence[int]], 
                                     grads: Union[torch.Tensor,Sequence[torch.Tensor]], 
                                     losses: Optional[Union[float,Sequence]] = None) -> tuple:
        """
        Preprocesses the gradients and losses before applying momentum.
        Recommended to be used in the `update_gradient` method.
        
        Args:
            indexes (Union[int,Sequence[int]]): The indexes of the gradient vectors and losses to be updated.
                The momentum with the given indexes will be updated based on the given gradients.
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update. 
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients. 
                The losses will be passed to base gradient operator. If the base gradient operator doesn't require loss information,
                you can set this value as None. Defaults to None.

        Returns:
            Tuple: A tuple containing the preprocessed indexes, gradients, and losses (if provided).
        """
        if not isinstance(indexes,Sequence):
            indexes=[indexes]
        if isinstance(grads,torch.Tensor):
            grads=[grads]
        if losses is not None:
            if self.loss_recorder is None:
                raise ValueError("Losses provided but loss_recorder is not provided.")
            if not isinstance(losses,Sequence):
                losses=[losses]
            losses=self.loss_recorder.record_loss(indexes,losses)
        if self.len_vectors is None or self.device is None:
            self.len_vectors = grads[0].shape[0]
            self.device = grads[0].device
        return indexes,grads,losses
        

    def update_gradient(self, network: torch.nn.Module, 
                        indexes: Union[int,Sequence[int]], 
                        grads: Union[torch.Tensor,Sequence[torch.Tensor]],
                        losses: Optional[Union[float,Sequence]] = None) -> None:
        """
        Updates the gradient of the given network with the calculated gradient.

        Args:
            network (torch.nn.Module): The network to update the gradient.
            indexes (Union[int,Sequence[int]]): The indexes of the gradient vectors and losses to be updated.
                The momentum with the given indexes will be updated based on the given gradients.
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update. 
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients. 
                The losses will be passed to base gradient operator. If the base gradient operator doesn't require loss information,
                you can set this value as None. Defaults to None.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
            
        Returns:
            None
        """
        apply_gradient_vector(network, self.calculate_gradient(indexes, grads, losses))


class PseudoMomentumOperator(MomentumOperator):

    """
    The major momentum version. 
    In this operator, the second momentum is estimated by a pseudo gradient based on the result of the gradient operator.
    NOTE: The momentum-based operator, e.g., Adam, is not recommend when using this operator. Please consider using SGD optimizer.
    
    Args:
        num_vectors (int): The number of gradient vectors.
        beta_1 (float): The moving average constant for the first momentum.
        beta_2 (float): The moving average constant for the second momentum.
        gradient_operator (GradientOperator, optional): The base gradient operator. Defaults to ConFIGOperator().
        loss_recorder (LossRecorder, optional): The loss recorder object. 
            If you want to pass loss information to "update_gradient" method or "apply_gradient" method, you need to specify a loss recorder. Defaults to None.

    Methods:
        calculate_gradient(indexes, grads, losses=None):
            Calculates the gradient based on the given indexes, gradients, and losses.
        update_gradient(network, indexes, grads, losses=None):
            Updates the gradient of the given network with the calculated gradient.
            
    Examples:
    ```python
    from conflictfree.momentum_operator import PseudoMomentumOperator
    from conflictfree.utils import get_gradient_vector,apply_gradient_vector
    optimizer=torch.Adam(network.parameters(),lr=1e-3)
    operator=PseudoMomentumOperator(num_vector=len(loss_fns)) # initialize operator, the only difference here is we need to specify the number of gradient vectors.
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
                num_vectors: int, 
                 beta_1: float=0.9, 
                 beta_2: float=0.999,
                 gradient_operator: GradientOperator=ConFIGOperator(),
                 loss_recorder: Optional[LossRecorder] = None) -> None:
        super().__init__(num_vectors, beta_1, beta_2, gradient_operator, loss_recorder)
        self.m=None
        self.s=None
        self.fake_m=None
        self.t=0
        self.t_grads=[0]*self.num_vectors
        self.all_initialized=False        
    
    def _preprocess_gradients_losses(self,
                                     indexes: Union[int,Sequence[int]], 
                                     grads: Union[torch.Tensor,Sequence[torch.Tensor]], 
                                     losses: Optional[Union[float,Sequence]] = None):
        indexes,grads,losses=super()._preprocess_gradients_losses(indexes, grads, losses)
        if self.m is None or self.s is None or self.fake_m is None:
            self.m=[torch.zeros(self.len_vectors,device=self.device) for i in range(self.num_vectors)]
            self.s=torch.zeros(self.len_vectors,device=self.device)
            self.fake_m=torch.zeros(self.len_vectors,device=self.device)        
        return indexes,grads,losses
    
    def calculate_gradient(self, 
                           indexes: Union[int,Sequence[int]], 
                           grads: Union[torch.Tensor,Sequence[torch.Tensor]], 
                           losses: Optional[Union[float,Sequence]] = None)->torch.Tensor:
        """
        Calculates the gradient based on the given indexes, gradients, and losses.

        Args:
            indexes (Union[int,Sequence[int]]): The indexes of the gradient vectors and losses to be updated.
                The momentum with the given indexes will be updated based on the given gradients.
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update. 
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients. 
                The losses will be passed to base gradient operator. If the base gradient operator doesn't require loss information,
                you can set this value as None. Defaults to None.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
            
        Returns:
            torch.Tensor: The calculated gradient.
        """
        with torch.no_grad():
            indexes,grads,losses=self._preprocess_gradients_losses(indexes,grads,losses)
            for i in range(len(indexes)):
                self.t_grads[indexes[i]]+=1
                self.m[indexes[i]]=self.beta_1*self.m[indexes[i]]+(1-self.beta_1)*grads[i]
            if not self.all_initialized:
                if has_zero(self.t_grads):
                    return torch.zeros_like(self.s)
                else:
                    self.all_initialized=True
            self.t+=1
            m_hats=torch.stack([self.m[i]/(1-self.beta_1**self.t_grads[i]) for i in range(self.num_vectors)],dim=0)
            final_grad=self.gradient_operator.calculate_gradient(m_hats,
                                                                 losses
                                                                 )
            fake_m=final_grad*(1-self.beta_1**self.t)
            fake_grad=(fake_m-self.beta_1*self.fake_m)/(1-self.beta_1)
            self.fake_m=fake_m
            self.s=self.beta_2*self.s+(1-self.beta_2)*fake_grad**2
            s_hat=self.s/(1-self.beta_2**self.t)
            final_grad=final_grad/(torch.sqrt(s_hat)+1e-8)
        return final_grad
    
class SeparateMomentumOperator(MomentumOperator):
    
    """
    In this operator, each gradient has its own second gradient. The gradient operator is applied on the rescaled momentum.
    NOTE: Please consider using the PseudoMomentumOperator since this operator does not give good performance according to our research.
    The momentum-based operator, e.g., Adam, is not recommend when using this operator. Please consider using SGD optimizer.
    
    Args:
        num_vectors (int): The number of gradient vectors.
        beta_1 (float): The moving average constant for the first momentum.
        beta_2 (float): The moving average constant for the second momentum.
        gradient_operator (GradientOperator, optional): The base gradient operator. Defaults to ConFIGOperator().
        loss_recorder (LossRecorder, optional): The loss recorder object. 
            If you want to pass loss information to "update_gradient" method or "apply_gradient" method, you need to specify a loss recorder. Defaults to None.

    Methods:
        calculate_gradient(indexes, grads, losses=None):
            Calculates the gradient based on the given indexes, gradients, and losses.
        update_gradient(network, indexes, grads, losses=None):
            Updates the gradient of the given network with the calculated gradient.
            
    """
    
    def __init__(self, 
                num_vectors: int, 
                 beta_1: float=0.9, 
                 beta_2: float=0.999,
                 gradient_operator: GradientOperator=ConFIGOperator(),
                 loss_recorder: Optional[LossRecorder] = None) -> None:
        super().__init__(num_vectors, beta_1, beta_2, gradient_operator, loss_recorder)
        self.m=None
        self.s=None
        self.t_grads=[0]*len(self.num_vectors)
        self.all_initialized=False        
        
    def _preprocess_gradients_losses(self,
                                     indexes: Union[int,Sequence[int]], 
                                     grads: Union[torch.Tensor,Sequence[torch.Tensor]], 
                                     losses: Optional[Union[float,Sequence]] = None):
        indexes,grads,losses=super()._preprocess_gradients_losses(indexes, grads, losses)
        if self.m is None or self.s is None:
            self.m=[torch.zeros(self.len_vectors,device=self.device) for i in range(self.num_vectors)]
            self.s=[torch.zeros(self.len_vectors,device=self.device) for i in range(self.num_vectors)]   
        return indexes,grads,losses
    
    def calculate_gradient(self, 
                           indexes: Union[int,Sequence[int]], 
                           grads: Union[torch.Tensor,Sequence[torch.Tensor]], 
                           losses: Optional[Union[float,Sequence]] = None)->torch.Tensor:
        """
        Calculates the gradient based on the given indexes, gradients, and losses.

        Args:
            indexes (Union[int,Sequence[int]]): The indexes of the gradient vectors and losses to be updated.
                The momentum with the given indexes will be updated based on the given gradients.
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update. 
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients. 
                The losses will be passed to base gradient operator. If the base gradient operator doesn't require loss information,
                you can set this value as None. Defaults to None.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
            
        Returns:
            torch.Tensor: The calculated gradient.
        """
        with torch.no_grad():
            indexes,grads,losses=self._preprocess_gradients_losses(indexes,grads,losses)
            for i in range(len(indexes)):
                self.t_grads[indexes[i]]+=1
                self.m[indexes[i]]=self.beta_1*self.m[indexes[i]]+(1-self.beta_1)*grads[i]
                self.s[indexes[i]]=self.beta_2*self.s[indexes[i]]+(1-self.beta_2)*grads[i]**2
            if not self.all_initialized:
                if has_zero(self.t_grads):
                    return torch.zeros_like(self.s)
                else:
                    self.all_initialized=True
            m_hats=torch.stack([self.m[i]/(1-self.betas_1**self.t_grads[i]) for i in range(self.num_vectors)],dim=0)
            s_hats=torch.stack([self.s[i]/(1-self.betas_2**self.t_grads[i]) for i in range(self.num_vectors)],dim=0)
        return self.gradient_operator.calculate_gradient(m_hats/(torch.sqrt(s_hats)+1e-8),
                                                         losses,)