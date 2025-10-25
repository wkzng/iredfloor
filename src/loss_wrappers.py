from abc import ABC, abstractmethod
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np



class TheoreticalLossWrapper(ABC):
    """
    Abstract base class for loss functions with theoretical bounds analysis.

    Provides:
    - Standard loss computation (forward) which returns both mean and variance of sample losses.
    - Instantaneous rate computation: RMS of gradients w.r.t. model outputs
    - Shape function: loss-specific theoretical properties
    """

    def __init__(self, loss_class, **loss_kwargs):
        """
        Args:
            loss_class: The PyTorch loss function class (e.g., nn.CrossEntropyLoss)
            **loss_kwargs: Arguments for the loss function (e.g., label_smoothing)
        """
        # Instantiate the base loss function with reduction='none' to get per-sample losses.
        # This is necessary to calculate the variance across the batch.
        self.per_sample_loss_fn = loss_class(reduction='none', **loss_kwargs)

    def forward(self, predictions:torch.Tensor, targets:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the task loss (mean over batch) and the variance of the sample losses.

        Returns:
            A tuple containing:
            - task_loss (torch.Tensor): The mean loss over the batch.
            - loss_variance (torch.Tensor): The variance of the losses over the batch.
        """
        # Calculate the loss for each sample in the batch
        per_sample_losses = self.per_sample_loss_fn(predictions, targets)

        # Ensure the tensor is float for variance calculation, if not already
        if not torch.is_floating_point(per_sample_losses):
            per_sample_losses = per_sample_losses.float()

        # Compute the mean (the primary task loss) and the variance
        loss_average = torch.mean(per_sample_losses)
        loss_variance = torch.var(per_sample_losses)
        return loss_average, loss_variance

    def __call__(self, predictions:torch.Tensor, targets:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Make the wrapper callable like a standard loss function"""
        return self.forward(predictions, targets)

    @abstractmethod
    def g_shape(self, loss_value:float) -> float:
        """ Compute g(L) such that  ∥∇L∥^2 ~ 1/D * λ(t) * g(L) """
        pass

    @abstractmethod
    def g_curv(self, loss_value:float) -> float:
        """ Compute second derivative of the shape function"""
        pass

    @torch.no_grad()
    def compute_residual_squared(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """ Compute the total squared residual error r^2 = sum_{i} ri^2
            with ri = yi - f(xi) where f(xi) is the model output
        """
        return F.mse_loss(prediction, target, reduction='sum')
    


class CrossEntropyLossWrapper(TheoreticalLossWrapper):
    def __init__(self, label_smoothing:float=0.0, alpha:float=3/2., n_class:int=None):
        super().__init__(
            loss_class=nn.CrossEntropyLoss,
            label_smoothing=label_smoothing
        )
        self.alpha = alpha
        self.n_class = n_class

    @torch.no_grad()
    def g_shape(self, loss_value:float) -> float:
        """ Curvature function for cross-entropy loss.
            ∥∇L∥^2 = 1/D * λ(t) * g(L)
            g(L) = alpha * (1 - exp(-L))^2
        """
        return self.alpha * (1 - np.exp(-loss_value))**2

    @torch.no_grad()
    def g_curv(self, loss_value:float) -> float:
        """ Compute g"(L) = alpha * 2 * exp(-L) * [2 * exp(-L) - 1 ] """
        exp_m_L = np.exp(-loss_value)
        return self.alpha * 2 * exp_m_L * (2 * exp_m_L - 1)
    
    @torch.no_grad()
    def compute_residual_squared(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """Compute the total squared residual error r^2 = sum_{i} ri^2"""
        probs = F.softmax(prediction, dim=-1)
        onehot = F.one_hot(target, num_classes=self.n_class)
        return F.mse_loss(probs, onehot, reduction='sum')
    


class MSELossWrapper(TheoreticalLossWrapper):
    def __init__(self):
        super().__init__(loss_class=nn.MSELoss)

    @torch.no_grad()
    def g_shape(self, loss_value:float) -> float:
        """ curvature function for MSE loss.
            ∥∇L∥^2 = 1/D * λ(t) * g(L)
            g(L) = 2 * L
        """
        return 2 * loss_value
    
    @torch.no_grad()
    def g_curv(self, loss_value:float) -> float:
        """ Compute g"(L) """
        return 0



class MAELossWrapper(TheoreticalLossWrapper):
    def __init__(self):
        super().__init__(loss_class=nn.L1Loss)

    @torch.no_grad()
    def g_shape(self, loss_value:float) -> float:
        """ curvature function for MAE loss.
            ∥∇L∥^2 = 1/D * λ(t) * g(L)
            g(L) = 1
        """
        return 1.0
    
    @torch.no_grad()
    def g_curv(self, loss_value:float) -> float:
        """ Compute g"(L) """
        return 0


class BCELossWrapper(TheoreticalLossWrapper):
    def __init__(self, n_class:int=None):
        super().__init__(loss_class=nn.BCELoss)
        self.n_class = n_class

    @torch.no_grad()
    def g_shape(self, loss_value:float) -> float:
        """ Shape function for BCE loss.,
            For BCE: L = 1/2 * sum_{k=1}^{n} y_k * log(p_k) + (1-y_k) * log(1- p_k)
        """
        raise NotImplementedError("TODO: derive theorecial expression")
    
    @torch.no_grad()
    def g_curv(self, loss_value:float) -> float:
        raise NotImplementedError("TODO: derive theorecial expression")

    @torch.no_grad()
    def compute_residual_squared(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """Compute the total squared residual error r^2 = sum_{i} ri^2"""
        #probs = torch.sigmoid(prediction)
        #onehot = F.one_hot(target, num_classes=self.n_class)
        #return F.mse_loss(probs, onehot, reduction='sum')
        raise NotImplementedError("TODO: derive theorecial expression")
    
