from . import *
class LossRecorder:
    
    def __init__(self,num_losses:int) -> None:
        self.num_losses=num_losses
    
    def record_loss(self,
                    losses_indexes:Union[int,Sequence[int]], 
                    losses: Union[float,Sequence]) -> None:
            """
            Records the given loss and returns the recorded loss.

            Args:
                losses (torch.Tensor): The loss to record.
                losses_index: The index of the loss.

            Returns:
                None

            Raises:
                NotImplementedError: If the method is not implemented.

            """
            raise NotImplementedError("record_loss method must be implemented")
    
    def record_all_losses(self,losses: Sequence) -> None:
        """
        Records all the losses and returns the recorded losses.

        Args:
            losses (torch.Tensor): The losses to record.

        Returns:
            None

        """
        if len(losses)!=self.num_losses:
            raise ValueError("The number of losses does not match the number of losses to be recorded.")
        return self.record_loss([i for i in range(self.num_losses)],losses)
    
    def _preprocess_losses(self,
                           losses_indexes: Union[int, Sequence[int]],
                           losses: Union[float, Sequence]):
        """
        Preprocesses the losses and their indexes. Recommended to be used in the `record_loss` method.

        Args:
            losses_indexes (Union[int, Sequence[int]]): The indexes of the losses.
            losses (Union[float, Sequence]): The losses.

        Returns:
            Tuple[Sequence[int], Sequence]: A tuple containing the preprocessed losses indexes and losses.
        """
        if isinstance(losses_indexes, int):
            losses_indexes = [losses_indexes]
        if isinstance(losses, float):
            losses = [losses]
        return losses_indexes, losses
    
class LatestLossRecorder(LossRecorder):
    """
    A loss recorder return the latest losses.
    """
    
    def __init__(self,num_losses:int) -> None:
        super().__init__(num_losses)
        self.losses=[None for i in range(num_losses)]
    
    def record_loss(self,
                    losses_indexes:Union[int,Sequence[int]], 
                    losses: Union[float,Sequence]) -> None:
        """
        Records the given loss and returns the recorded loss.

        Args:
            losses (Sequence): The loss to record.
            losses_index: The index of the loss.

        Returns:
            None

        """
        losses_indexes,losses=self._preprocess_losses(losses_indexes,losses)
        for i in losses_indexes:
            self.losses[i]=losses[losses_indexes.index(i)]
        return self.losses
    
class MomentumLossRecorder(LossRecorder):
    """
    A loss recorder that records the momentum of the loss.
    """
    
    def __init__(self, num_losses:int, betas: Union[float, Sequence[float]] = 0.9):
        super().__init__(num_losses)
        if isinstance(betas, float):
            self.betas = [betas] * num_losses
        self.m=[0.0 for i in range(num_losses)]
        self.t=[0 for i in range(num_losses)]
    
    def record_loss(self,
                    losses_indexes:Union[int,Sequence[int]], 
                    losses: Union[float,Sequence]) -> None:
        """
        Records the given loss and returns the recorded loss.

        Args:
            losses (Sequence): The loss to record.
            losses_index: The index of the loss.

        Returns:
            None

        """
        losses_indexes,losses=self._preprocess_losses(losses_indexes,losses)
        for index in losses_indexes:
            self.t[index]+=1
            self.m[index]=self.betas*self.m[index]+(1-self.betas[index])*losses[losses_indexes.index(index)]
        return [self.m[index]/(1-self.betas[index]**self.t[index]) for index in len(self.m)]