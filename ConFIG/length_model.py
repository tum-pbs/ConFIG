from . import *
from .utils import *


class LengthModel:
    """
    This class represents a length model.

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
                   losses:Optional[Sequence]=None):
        """
        Calculates the length based on the given parameters.

        Args:
            target_vector: The target vector.
            unit_target_vector: The unit target vector.
            gradients: The gradients.
            losses: The losses.

        Returns:
            The calculated length.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")
    
    def rescale_length(self, 
                       target_vector:torch.Tensor,
                       gradients:Optional[torch.Tensor]=None,
                       losses:Optional[Sequence]=None):
        """
        Rescales the length of the target vector based on the given parameters.

        Args:
            target_vector: The target vector.
            gradients: The gradients.
            losses: The losses.

        Returns:
            The rescaled length.
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
                         losses:Optional[Sequence]=None):
        if gradients is None:
            raise ValueError("The ProjectLength model requires gradients information.")
        return torch.sum(torch.stack([torch.dot(grad_i,unit_target_vector) for grad_i in gradients]))