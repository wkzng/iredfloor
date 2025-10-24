from abc import ABC, abstractmethod
from torch import nn
import numpy as np



class TheoreticalLossWrapper(ABC):
    """
    Abstract base class for loss functions with theoretical bounds analysis.

    Provides:
    - Standard loss computation (forward)
    - Instantaneous rate computation: RMS of gradients w.r.t. model outputs
    - Shape function: loss-specific theoretical properties
    """

    def __init__(self, base_loss_fn):
        """
        Args:
            base_loss_fn: The underlying PyTorch loss function (e.g., nn.CrossEntropyLoss())
        """
        self.base_loss_fn = base_loss_fn

    def forward(self, predictions, targets):
        """Standard loss computation - delegates to base loss function"""
        return self.base_loss_fn(predictions, targets)

    def __call__(self, predictions, targets):
        """Make the wrapper callable like a standard loss function"""
        return self.forward(predictions, targets)

    @abstractmethod
    def curvature_squared(self, loss_value:float) -> float:
        """ Compute F(L)^2 such that  ∥∇L∥^2 ~ 1/D * λ(t) * F(L)^2 """
        pass



class CrossEntropyLossWrapper(TheoreticalLossWrapper):
    def __init__(self, reduction='mean', label_smoothing:float=0.0, alpha:float=3/2.):
        super().__init__(
            base_loss_fn=nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
        )
        self.alpha = alpha

    def curvature_squared(self, loss_value:float) -> float:
        """ Curvature function for cross-entropy loss.
            ∥∇L∥^2 = 1/D * λ(t) * F(L)^2
            F(L)^2 = alpha * (1 - exp(-L))^2
        """
        return self.alpha * (1 - np.exp(-loss_value))**2


class MSELossWrapper(TheoreticalLossWrapper):
    def __init__(self, reduction='mean'):
        super().__init__(
            base_loss_fn=nn.MSELoss(reduction=reduction)
        )

    def curvature_squared(self, loss_value:float) -> float:
        """ curvature function for MSE loss.
            ∥∇L∥^2 = 1/D * λ(t) * F(L)^2
            F(L)^2 = 2 * L
        """
        return 2 * loss_value


class MAELossWrapper(TheoreticalLossWrapper):
    def __init__(self, reduction='mean'):
        super().__init__(
            base_loss_fn=nn.L1Loss(reduction=reduction)
        )

    def curvature_squared(self, loss_value:float) -> float:
        """ curvature function for MAE loss.
            ∥∇L∥^2 = 1/D * λ(t) * F(L)^2
            F(L)^2 = 1
        """
        return 1.0


class BCELossWrapper(TheoreticalLossWrapper):
    def __init__(self, reduction='mean'):
        super().__init__(
            base_loss_fn=nn.BCELoss(reduction=reduction)
        )

    def curvature_squared(self, loss_value:float) -> float:
        """ Shape function for BCE loss.
            For BCE: L = 1/2 * sum_{k=1}^{n} y_k * log(p_k) + (1-y_k) * log(1- p_k)
        """
        raise NotImplementedError("TODO: derive theorecial expression")
