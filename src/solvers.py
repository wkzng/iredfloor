from scipy.interpolate import make_interp_spline
from scipy.integrate import odeint
import numpy as np


from src.loss_wrappers import TheoreticalLossWrapper


class LossDynamicsSolver:
    def __init__(self, 
            loss_fn_wrapper:TheoreticalLossWrapper, 
            initial_loss:float, 
            time_steps:float|np.ndarray,
            rates:float|np.ndarray,
            batch_sizes: float|np.ndarray,
        ):
        """
            loss_fn_wrapper: wrapper for the loss function along with the curvatuve functions
            initial_loss : initial value of the training loss across batche or the entire dataset
            learning_rate: sequence of learning rate (this can be a constant value if the learning rate is fixed)
            rate: sequence of convergence rates
            batch_sizes: seauence of batch sizes
        """
        self.loss_fn_wrapper = loss_fn_wrapper
        self.initial_loss = initial_loss
        self.time_steps = time_steps
        self.rates = rates
        self.batch_sizes = batch_sizes
        self.rate_interpolator = make_interp_spline(self.time_steps, self.rates, k=3) # k=3 (cubic), s=0 forces exact fit
        self.batch_interpolator = make_interp_spline(self.time_steps, self.batch_sizes, k=3) # k=3 (cubic), s=0 forces exact fit


    def solve(self):
        """ Compute numerical solutions to:  dL/dt = - lamba_eff(t) / batch_size(t) * F(L)^2 
            with
                - lamba_eff(t): as the effective rate of decay
                - batch_size(t): the record of batch size
        """
        dx_dt = lambda x, t : - self.rate_interpolator(t) * self.loss_fn_wrapper.g_shape(x) / self.batch_interpolator(t)
        return odeint(dx_dt, self.initial_loss, self.time_steps)
