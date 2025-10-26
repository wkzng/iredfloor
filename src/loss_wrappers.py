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
        loss_variance = torch.var(per_sample_losses, unbiased=False)
        return loss_average, loss_variance

    def __call__(self, predictions:torch.Tensor, targets:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Make the wrapper callable like a standard loss function"""
        return self.forward(predictions, targets)

    @abstractmethod
    def q_value(self, loss_value:float) -> float:
        """ Compute topology function"""
        pass

    @abstractmethod
    def q_second_derivative(self, loss_value:float) -> float:
        """ Compute second derivative of the toppoly function"""
        pass

    def g_value(self, loss_value:float, variance:float) -> float:
        """Default 2^n order closure: q(L) + 0.5 * Var(L_i) * q''(L)."""
        return self.q_value(loss_value) + 0.5 * variance * self.q_second_derivative(loss_value)

    @torch.no_grad()
    def compute_residual_squared(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """ Compute the total squared residual error r^2 = sum_{i} ri^2
            with ri = yi - f(xi) where f(xi) is the model output
        """
        return F.mse_loss(prediction.detach(), target, reduction='sum')
    


class CrossEntropyLossWrapper(TheoreticalLossWrapper):
    def __init__(self, n_class:int, label_smoothing:float=0.0):
        super().__init__(
            loss_class=nn.CrossEntropyLoss,
            label_smoothing=label_smoothing
        )
        assert n_class > 1
        self.alpha = 0.5 * (3 + 1/(n_class - 1)) # midpoint of the admissible interval
        self.n_class = n_class

    @torch.no_grad()
    def q_value(self, loss_value:float) -> float:
        """ q(L) = alpha * (1 - exp(-L))^2 """
        return self.alpha * (1 - np.exp(-loss_value))**2

    @torch.no_grad()
    def q_second_derivative(self, loss_value:float) -> float:
        """ Compute q"(L) = alpha * 2 * exp(-L) * [2 * exp(-L) - 1 ] """
        exp_m_L = np.exp(-loss_value)
        return self.alpha * 2 * exp_m_L * (2 * exp_m_L - 1)
    
    @torch.no_grad()
    def compute_residual_squared(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """Compute the total squared residual error r^2 = sum_{i} ri^2"""
        probs = F.softmax(prediction.detach(), dim=-1)
        C = self.n_class or prediction.size(-1)
        y = F.one_hot(target, num_classes=C).to(probs.dtype).to(probs.device)

        if self.per_sample_loss_fn.label_smoothing > 0:
            # smoothed labels: (1-ε) on true class, ε/C elsewhere
            eps = self.per_sample_loss_fn.label_smoothing
            y = (1 - eps) * y + eps / C

        return F.mse_loss(probs, y, reduction='sum')
    


class MSELossWrapper(TheoreticalLossWrapper):
    def __init__(self):
        super().__init__(loss_class=nn.MSELoss)


    def forward(self, predictions:torch.Tensor, targets:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Calculate the loss for each sample in the batch
        per_sample_losses = self.per_sample_loss_fn(predictions, targets) # shape [B, ...]
    
        while per_sample_losses.dim() > 1:
            per_sample_losses = per_sample_losses.mean(dim=-1)

        loss_average = per_sample_losses.mean()
        loss_variance = per_sample_losses.var(unbiased=False)
        return loss_average, loss_variance
    
    @torch.no_grad()
    def q_value(self, loss_value:float) -> float:
        """ topology function for MSE loss: q(L) = 2 * L"""
        return 2 * loss_value
    
    @torch.no_grad()
    def q_second_derivative(self, loss_value:float) -> float:
        """ Compute q"(L) """
        return 0



class MAELossWrapper(TheoreticalLossWrapper):
    def __init__(self):
        super().__init__(loss_class=nn.L1Loss)

    @torch.no_grad()
    def q_value(self, loss_value:float) -> float:
        """ curvature function for MAE loss: q(L) = 1"""
        return 1.0
    
    @torch.no_grad()
    def q_second_derivative(self, loss_value:float) -> float:
        """ Compute q"(L) """
        return 0


class BCELossWrapper(TheoreticalLossWrapper):
    def __init__(self, n_class:int=None):
        super().__init__(loss_class=nn.BCELoss)
        self.n_class = n_class

    @torch.no_grad()
    def q_value(self, loss_value:float) -> float:
        """ Topology function for BCE loss.,
            For BCE: L = 1/2 * sum_{k=1}^{n} y_k * log(p_k) + (1-y_k) * log(1- p_k)
        """
        raise NotImplementedError("TODO: derive theorecial expression")
    
    @torch.no_grad()
    def q_second_derivative(self, loss_value:float) -> float:
        raise NotImplementedError("TODO: derive theorecial expression")

    @torch.no_grad()
    def compute_residual_squared(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """Compute the total squared residual error r^2 = sum_{i} ri^2"""
        #probs = torch.sigmoid(prediction)
        #onehot = F.one_hot(target, num_classes=self.n_class)
        #return F.mse_loss(probs, onehot, reduction='sum')
        raise NotImplementedError("TODO: derive theorecial expression")
    
