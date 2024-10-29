from torch import Tensor
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

    def get_length(
        self,
        target_vector: Optional[torch.Tensor] = None,
        unit_target_vector: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ) -> Union[torch.Tensor, float]:
        """
        Calculates the length based on the given parameters. Not all parameters are required.

        Args:
            target_vector (Optional[torch.Tensor]): The final update gradient vector.
            unit_target_vector (Optional[torch.Tensor]): The unit vector of the target vector.
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix. The shape of this tensor should be (m,N) where m is the number of gradients and N is the number of elements of each gradients.
            losses (Optional[Sequence]): The losses.

        Returns:
            Union[torch.Tensor, float]: The calculated length.
        """
        raise NotImplementedError(
            "This method must be implemented by the subclass.")

    def rescale_length(
        self,
        target_vector: torch.Tensor,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Rescales the length of the target vector based on the given parameters.
        It calls the get_length method to calculate the length and then rescales the target vector.

        Args:
            target_vector (torch.Tensor): The final update gradient vector.
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix. The shape of this tensor should be (m,N) where m is the number of gradients and N is the number of elements of each gradients.
            losses (Optional[Sequence]): The losses.

        Returns:
            torch.Tensor: The rescaled target vector.
        """
        unit_target_vector = unit_vector(target_vector)
        return (
            self.get_length(
                target_vector=target_vector,
                unit_target_vector=unit_target_vector,
                gradients=gradients,
                losses=losses,
            )
            * unit_target_vector
        )


class ProjectionLength(LengthModel):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector:
    
    $$
    |\mathbf{g}_c|=\sum_{i=1}^m|\mathbf{g}_i|\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$
    """

    def __init__(self):
        super().__init__()

    def get_length(
        self,
        target_vector: Optional[torch.Tensor] = None,
        unit_target_vector: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the length based on the given parameters. Not all parameters are required.

        Args:
            target_vector (Optional[torch.Tensor]): The final update gradient vector.
                One of the `target_vector` or `unit_target_vector` parameter need to be provided.
            unit_target_vector (Optional[torch.Tensor]): The unit vector of the target vector.
                One of the `target_vector` or `unit_target_vector` parameter need to be provided.
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix. The shape of this tensor should be (m,N) where m is the number of gradients and N is the number of elements of each gradients.
            losses (Optional[Sequence]): The losses. Not used in this model.

        Returns:
            Union[torch.Tensor, float]: The calculated length.
        """
        assert gradients is not None, "The ProjectionLength model requires gradients information."
        if unit_target_vector is None:
            unit_target_vector = unit_vector(target_vector)
        return torch.sum(
            torch.stack([torch.dot(grad_i, unit_target_vector)
                        for grad_i in gradients])
        )


class _FlexibleTrackProjectionLength(LengthModel):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    The length each loss-specific gradient will be rescaled to the same length as the tracked value before projection.
    The tracked value is calculated by the _tracked_value method.
    """

    def __init__(self):
        super().__init__()

    def get_length(
        self,
        target_vector: Optional[torch.Tensor] = None,
        unit_target_vector: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the length based on the given parameters. Not all parameters are required.

        Args:
            target_vector (Optional[torch.Tensor]): The final update gradient vector.
                One of the `target_vector` or `unit_target_vector` parameter need to be provided.
            unit_target_vector (Optional[torch.Tensor]): The unit vector of the target vector.
                One of the `target_vector` or `unit_target_vector` parameter need to be provided.
            gradients (Optional[torch.Tensor]): The loss-specific gradients matrix. The shape of this tensor should be (m,N) where m is the number of gradients and N is the number of elements of each gradients.
            losses (Optional[Sequence]): The losses. Not used in this model.

        Returns:
            Union[torch.Tensor, float]: The calculated length.
        """
        assert gradients is not None, "The ProjectLength model requires gradients information."
        if unit_target_vector is None:
            unit_target_vector = unit_vector(target_vector)
        norms = torch.norm(gradients, dim=1)
        tracked_value = self._tracked_value(norms)
        return sum(
            [
                torch.dot(grad_i / norm_i, unit_target_vector) * tracked_value
                for grad_i, norm_i in zip(gradients, norms)
            ]
        )

    def _tracked_value(self, grad_norms: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "This method must be implemented by the subclass.")


class TrackMinimum(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the same length as the minimum gradient before projection, i.e., the minimum gradient will be the same length as the target vector.
    
    $$
    |\mathbf{g}_c|=\sum_{i=1}^m|\mathbf{g}_{min}|\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$
    """

    def __init__(self):
        super().__init__()

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return grad_norms.min()


class TrackMaximum(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the same length as the maximum gradient before projection, i.e., the maximum gradient will be the same length as the target vector.
    
    $$
    |\mathbf{g}_c|=\sum_{i=1}^m|\mathbf{g}_{max}|\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$
    """

    def __init__(self):
        super().__init__()

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return grad_norms.max()


class TrackHarmonicAverage(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the harmonic average of the lengths of all gradients before projection, i.e., the minimum gradient will be the same length as the target vector.
    
    $$
    |\mathbf{g}_c|=\sum_{i=1}^m\overline{|\mathbf{g}|}_{harm}\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$
    
    where
    
    $$
    \overline{|\mathbf{g}|}_{harm}=\frac{m}{\sum_{i=1}^m \frac{1}{|\mathbf{g}_i|}}
    $$
    
    The harmonic average can be used to avoid the influence of the large gradients.
    """

    def __init__(self):
        super().__init__()

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return grad_norms.shape[0] / torch.sum(1 / grad_norms)


class TrackArithmeticAverage(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the arithmetic average of the lengths of all gradients before projection, i.e., the minimum gradient will be the same length as the target vector.
    
    $$
    |\mathbf{g}_c|=\sum_{i=1}^m\overline{|\mathbf{g}|}_{arith}\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$
    
    where
    
    $$
    \overline{|\mathbf{g}|}_{arith}=\frac{1}{m}\sum_{i=1}^m |\mathbf{g}_i|
    $$
    """

    def __init__(self):
        super().__init__()

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return grad_norms.mean()


class TrackGeometricAverage(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the geometric average of the lengths of all gradients before projection, i.e., the minimum gradient will be the same length as the target vector.
    
    $$
    |\mathbf{g}_c|=\sum_{i=1}^m\overline{|\mathbf{g}|}_{geom}\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$
    
    where
    
    $$
    \overline{|\mathbf{g}|}_{geom}=\left(\prod_{i=1}^m |\mathbf{g}_i|\right)^{\frac{1}{m}}
    $$
    
    The geometric average can be used to avoid the influence of the large gradients.
    """

    def __init__(self):
        super().__init__()

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return torch.prod(grad_norms) ** (1 / grad_norms.shape[0])


class TrackSpecific(_FlexibleTrackProjectionLength):
    """
    Rescale the length of the target vector based on the projection of the gradients on the target vector.
    All the gradients will be rescaled to the same length as the specific gradient before projection.
    E.g., if the track_id is 2, then all the gradients will be rescaled to the same length as the third gradient before projection.
    
    $$
    |\mathbf{g}_c|=\sum_{i=1}^m\overline{|\mathbf{g}|}_{track_id}\mathcal{S}_c(\mathbf{g}_i,\mathbf{g}_c)
    $$

    """

    def __init__(self, track_id: int):
        super().__init__()
        self.track_id = track_id

    def _tracked_value(self, grad_norms: Tensor) -> Tensor:
        return grad_norms[self.track_id]
