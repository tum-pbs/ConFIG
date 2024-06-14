from . import *

class WeightModel:
    def __init__(self):
        pass
    
    def get_weights(self, gradients:Optional[torch.Tensor]=None,losses:Optional[Sequence]=None):
        raise NotImplementedError("This method must be implemented by the subclass.")

class EqualWeight(WeightModel):
    """
    A weight model that assigns equal weights to all gradients.
    """

    def __init__(self):
        super().__init__()
    
    def get_weights(self, gradients: torch.Tensor, losses: Optional[Sequence] = None,device:Optional[Union[torch.device,str]]=None) -> torch.Tensor:
        """
        Calculate the weights for the given gradients.

        Args:
            gradients (torch.Tensor): The gradients to calculate weights for.
            losses (Optional[Sequence]): Optional sequence of losses. Not used in this model.

        Returns:
            torch.Tensor: A tensor of equal weights for all gradients.
        
        Raises:
            ValueError: If gradients is None.
        """
        if gradients is None:
            raise ValueError("The EqualWeight model requires gradients.")
        return torch.ones(gradients.shape[0],device=device)