from . import *
from .utils import *


class LengthModel:
    """
    The base class for length model.

    Methods:
        get_length: Calculates the length based on the given parameters.
        rescale_length: Rescales the length of the target vector based on the given parameters.
    """
    def __init__(self):
        pass
    
    def get_length(self, 
                   target_vector:Optional[torch.Tensor]=None,
                   unit_target_vector:Optional[torch.Tensor]=None,
                   gradients:Optional[torch.Tensor]=None,
                   losses:Optional[Sequence]=None)-> Union[torch.Tensor, float]:
        """
        Calculates the length based on the given parameters. Not all parameters are required.

        Args:
            target_vector (Optional[torch.Tensor]): The final update gradient vector.
            unit_target_vector (Optional[torch.Tensor]): The unit vector of the target vector.
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix.
            losses (Optional[Sequence]): The losses.

        Returns:
            Union[torch.Tensor, float]: The calculated length.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")
    
    def rescale_length(self, 
                       target_vector:torch.Tensor,
                       gradients:Optional[torch.Tensor]=None,
                       losses:Optional[Sequence]=None)->torch.Tensor:
        """
        Rescales the length of the target vector based on the given parameters.
        It calls the get_length method to calculate the length and then rescales the target vector.
        
        Args:
            target_vector (torch.Tensor): The final update gradient vector.
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix.
            losses (Optional[Sequence]): The losses.
        
        Returns:
            torch.Tensor: The rescaled target vector.
        """
        unit_target_vector = unit_vector(target_vector)
        return self.get_length(target_vector=target_vector,
                               unit_target_vector=unit_target_vector,
                               gradients=gradients,
                               losses=losses) * unit_target_vector
    
class ProjectionLength(LengthModel):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    """

    def __init__(self):
        super().__init__()
        
    def get_length(self, target_vector:Optional[torch.Tensor]=None,
                         unit_target_vector:Optional[torch.Tensor]=None,
                         gradients:Optional[torch.Tensor]=None,
                         losses:Optional[Sequence]=None)->torch.Tensor:
        """
        Calculates the length based on the given parameters. Not all parameters are required.

        Args:
            target_vector (Optional[torch.Tensor]): The final update gradient vector. 
                One of the `target_vector` or `unit_target_vector` parameter need to be provided.
            unit_target_vector (Optional[torch.Tensor]): The unit vector of the target vector.
                One of the `target_vector` or `unit_target_vector` parameter need to be provided.
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix.
            losses (Optional[Sequence]): The losses. Not used in this model.

        Returns:
            Union[torch.Tensor, float]: The calculated length.
        """
        if gradients is None:
            raise ValueError("The ProjectLength model requires gradients information.")
        if unit_target_vector is None:
            unit_target_vector = unit_vector(target_vector)
        return torch.sum(torch.stack([torch.dot(grad_i,unit_target_vector) for grad_i in gradients]))