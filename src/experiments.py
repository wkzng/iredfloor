import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from abc import ABC, abstractmethod
from tqdm import tqdm

from torch.utils.data import DataLoader
from src.loss_wrappers import TheoreticalLossWrapper
from src.models import ModelCreator





class BaseExperiment(ABC):
    """Base class for testing optimization identities with efficient gradient computation"""

    def __init__(self, model_creator:ModelCreator, train_loader:DataLoader, test_loader:DataLoader,
                 learning_rate:float, weight_decay:float, task_loss_fn:TheoreticalLossWrapper, max_steps:int):
        self.model_creator = model_creator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.task_loss_fn = task_loss_fn
        self.max_steps = max_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler(device=str(self.device))
        self.n_dataset = len(self.train_loader.dataset)


    @abstractmethod
    def _create_optimizer(self, model:nn.Module) -> optim.Optimizer:
        """Create the specific optimizer for this tester"""
        pass


    def _compute_task_loss_only(self, model:nn.Module, data_loader):
        """Compute task loss over a dataset (no gradients)"""
        model.eval()
        total_task_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                task_loss, _ = self.task_loss_fn(logits, y)
                total_task_loss += task_loss * x.size(0)
                total_samples += x.size(0)

        model.train()
        return total_task_loss.item() / total_samples


    def _compute_weight_norm_sq(self, model:nn.Module) -> float:
        """Compute squared norm of model weights"""
        return sum(p.norm(2)**2 for p in model.parameters()).item()


    def _compute_regularization(self, weights_norm_sq:float) -> float:
        """Compute regularization term"""
        return 0.5 * self.weight_decay * weights_norm_sq


    def _compute_batch_metrics(
            self, model:nn.Module, task_loss:torch.Tensor, weights_norm_sq:float, batch_size:int,
            L_task_variance:torch.Tensor, task_residual_error_sq:torch.Tensor
        ) -> dict:
        """Compute all metrics for current training batch"""
        # Task loss (pure, from batch)
        L_task_batch_t = task_loss.detach().item()
        L_task_variance = L_task_variance.detach().item()
        task_residual_error_sq = task_residual_error_sq.detach().item()

        # Regularization
        L_reg_batch_t = self._compute_regularization(weights_norm_sq)

        # Full loss = task + regularization
        L_full_batch_t = L_task_batch_t + L_reg_batch_t

        # Task gradient metrics || grad L_task ||^2 (from .grad fields after backward())
        # This is the PURE task gradient, as .step() has not been called.
        # grad_L_task_norm_sq_t = sum(p.grad.norm(2)**2 for p in model.parameters() if p.grad is not None).item()
        grad_L_task_norm_sq_t = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_L_task_norm_sq_t += torch.dot(p.grad.flatten(), p.grad.flatten())
        grad_L_task_norm_sq_t = grad_L_task_norm_sq_t.item()

        # Shape function g(L_task) and its approximations
        g_rayleigh = task_residual_error_sq / batch_size
        g_approx_1st_order = self.task_loss_fn.g_shape(L_task_batch_t)
        g_approx_curvature = self.task_loss_fn.g_curv(L_task_batch_t)
        g_approx_2nd_order = g_approx_1st_order + 0.5 * L_task_variance * g_approx_curvature

        eps = 1e-12
        g_approx_1st_order = max(g_approx_1st_order, eps)
        g_approx_2nd_order = max(g_approx_2nd_order, eps)
        g_rayleigh = max(g_rayleigh, eps)

        # Effective eigenvalue lambda_eff(t)
        lambda_rayleigh = batch_size * grad_L_task_norm_sq_t / g_rayleigh
        lambda_approx_1st_order = batch_size * grad_L_task_norm_sq_t / g_approx_1st_order
        lambda_approx_2nd_order = batch_size * grad_L_task_norm_sq_t / g_approx_2nd_order

        # gap between approximation of lambda
        lambda_gap_1st_order = lambda_rayleigh - lambda_approx_1st_order
        lambda_gap_2nd_order = lambda_rayleigh - lambda_approx_2nd_order

        lambda_gap_rel_1st =  lambda_gap_1st_order / (lambda_rayleigh + eps)
        lambda_gap_rel_2nd =  lambda_gap_2nd_order / (lambda_rayleigh + eps)

        return {
            'L_task': L_task_batch_t,
            'L_reg': L_reg_batch_t,
            'L_full': L_full_batch_t,
            '||w||^2': weights_norm_sq,
            '||gradL_task||^2': grad_L_task_norm_sq_t,
            "variance[L_task]": L_task_variance,

            "g_approx_curvature": g_approx_curvature,
            'g_approx_1st_order': g_approx_1st_order,
            'g_approx_2nd_order': g_approx_2nd_order,
            'g_rayleigh': g_rayleigh,

            'lambda_approx_1st_order': lambda_approx_1st_order,
            'lambda_approx_2nd_order': lambda_approx_2nd_order,
            "lambda_rayleigh": lambda_rayleigh,

            "lambda_gap_1st_order": lambda_gap_1st_order,
            "lambda_gap_2nd_order": lambda_gap_2nd_order,

            "lambda_gap_rel_1st": lambda_gap_rel_1st,
            "lambda_gap_rel_2nd": lambda_gap_rel_2nd,

            'n_dataset': self.n_dataset,
            "batch_size": batch_size
        }

    def _compute_eval_test_losses(self, weights_norm_sq:float, model:nn.Module, step:int, eval_frequency:int) -> dict:
        """Compute eval and test losses periodically for analysis"""
        eval_test_metrics = {}

        if step % eval_frequency == 0:
            eval_test_metrics = {
                'L_task_test_t': self._compute_task_loss_only(model, self.test_loader),
            }
        return eval_test_metrics


    def run(self, eval_frequency:int=None):
        """ Run optimizer testing and return results as pandas DataFrame
            Args:
                eval_frequency: How often to compute eval/test losses (default every 50 steps)
        """
        # Initialize model and optimizer
        model = self.model_creator.create().to(self.device)
        optimizer = self._create_optimizer(model)

        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=4, mode="triangular")
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.95)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=self.max_steps)

        #set test and eval set test frequency
        eval_frequency = len(self.train_loader) if eval_frequency is None else eval_frequency
        eval_frequency = max(eval_frequency, 100)

        # Storage for results
        results = []

        print(f"Running {self.__class__.__name__} test for {self.max_steps} steps | Evaluating every {eval_frequency} steps...")
        print("=" * 80)

        train_loader_iter = iter(self.train_loader)
        epoch = 0
        time_step = 0
        progress_bar = tqdm(range(self.max_steps), ncols=80, total=self.max_steps)

        for step in progress_bar:
            try:
                # Get next training batch
                x, y = next(train_loader_iter)
            except StopIteration:
                # Restart iterator if we've gone through all data
                train_loader_iter = iter(self.train_loader)
                x, y = next(train_loader_iter)
                epoch = epoch + 1
                scheduler.step()

            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.shape[0]

            # === FORWARD PASS ===
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            task_loss, L_task_variance = self.task_loss_fn.forward(logits, y)  # Pure task loss

            # optimizer.zero_grad(set_to_none=True)
            # with torch.cuda.amp.autocast():
            #     logits = model(x)
            #     task_loss, L_task_variance = self.task_loss_fn.forward(logits, y)

            # ==== COMPUTE THE RESIDUAL ====
            task_residual_error_sq = self.task_loss_fn.compute_residual_squared(logits, y)

            # === COMPUTE PRE-BACKWARD METRICS ===
            weights_norm_sq = self._compute_weight_norm_sq(model)
            learning_rate = scheduler.get_last_lr()[0]

            # === BACKWARD PASS ===
            task_loss.backward()
            # self.scaler.scale(task_loss).backward()
            # self.scaler.step(optimizer)
            # self.scaler.update()

            # === COMPUTE BATCH METRICS ===
            batch_metrics = self._compute_batch_metrics(
                model=model,
                task_loss=task_loss,
                weights_norm_sq=weights_norm_sq,
                batch_size=batch_size,
                L_task_variance=L_task_variance,
                task_residual_error_sq=task_residual_error_sq
            )

            # === COMPUTE EVAL/TEST LOSSES (PERIODIC) ===
            eval_test_metrics = self._compute_eval_test_losses(
                weights_norm_sq=weights_norm_sq,
                model=model,
                step=step,
                eval_frequency=eval_frequency
            )

            # === OPTIMIZER STEP ===
            optimizer.step()  # This modifies gradients internally, but we're done with them

            # === STORE RESULTS ===
            result = {
                'step': step,
                't': time_step,
                "learning_rate": learning_rate,
                "epoch": epoch,
                **batch_metrics,
                **eval_test_metrics,
            }
            results.append(result)
            time_step += learning_rate

            #update logs
            progress_bar.set_description(f"Epoch {epoch} | learning_rate={learning_rate} ")


        # Convert to DataFrame
        df = pd.DataFrame(results)
        print(f"✅ {self.__class__.__name__} testing completed! Collected {len(df)} steps of data.")
        return df




class SGDExperiment(BaseExperiment):
    """SGD-specific tester for the identity: L_full(t) = L_full(0) - ∫₀ᵗ γ(τ) ||∇L_full(τ)||² dτ"""
    name:str = "SGD"

    def _create_optimizer(self, model:nn.Module) -> optim.SGD:
        return optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def _compute_full_gradient_norm_sq(self, model:nn.Module, grad_task_norm_sq:float, weights_norm_sq:float) -> float:
        """
        Compute ||∇L_full||² = ||∇L_task + λθ||² without contaminating gradients
        Using: ||∇L_task + λθ||² = ||∇L_task||² + 2λ⟨∇L_task,θ⟩ + λ²||θ||²
        """
        if self.weight_decay == 0:
            return grad_task_norm_sq

        # Compute ⟨∇L_task, θ⟩ = dot product between task gradients and weights
        task_weight_dot_product = sum(
            (p.grad * p.data).sum().item()
            for p in model.parameters() if p.grad is not None
        )

        # Full gradient norm squared
        grad_full_norm_sq = (
            grad_task_norm_sq +
            2 * self.weight_decay * task_weight_dot_product +
            self.weight_decay**2 * weights_norm_sq
        )
        return grad_full_norm_sq


class AdaGradExperiment(BaseExperiment):
    name:str = "AdaGrad"

    def _create_optimizer(self, model:nn.Module) -> optim.Adagrad:
        return optim.Adagrad(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)





if __name__ == "__main__":
    from loss_wrappers import CrossEntropyLossWrapper
    from models import ModelCreator


    # Create tester
    max_steps = 1000 * 10
    architecture ="deepMLP" # "WideMLP"
    model_args = {"depth":3}

    model_creator = ModelCreator(architecture, model_args=model_args)
    model_creator.create()
    task_loss_fn = CrossEntropyLossWrapper()
    initial_learning_rate = 1e-3


    experiment = SGDExperiment(
        model_creator=model_creator,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=initial_learning_rate,
        weight_decay=initial_learning_rate/100,
        task_loss_fn=task_loss_fn,
        max_steps=max_steps
    )

    df:pd.DataFrame = experiment.run()

    file_name = f"experiment_{experiment.name}_{architecture}_{max_steps}.csv"
    df.to_csv(file_name, index=False)
    files.download(file_name)