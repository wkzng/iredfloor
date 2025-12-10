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
            vars:float|np.ndarray|None=None,
        ):
        """
            loss_fn_wrapper: wrapper for the loss function along with the curvatuve functions
            initial_loss : initial value of the training loss across batche or the entire dataset
            learning_rate: sequence of learning rate (this can be a constant value if the learning rate is fixed)
            rate: sequence of convergence rates
        """
        self.loss_fn_wrapper = loss_fn_wrapper
        self.initial_loss = initial_loss
        self.time_steps = time_steps
        vars = np.zeros_like(time_steps) if vars is None else vars
        self.phi_interpolator = make_interp_spline(self.time_steps, rates, k=3) # k=3 (cubic), s=0 forces exact fit
        self.var_interpolator = make_interp_spline(self.time_steps, vars, k=3) # k=3 (cubic), s=0 forces exact fit


    def solve(self):
        """ Compute numerical solutions to:  dL/dt = - phi(t) * g(L)
            with
                - lamba_eff(t): as the effective rate of decay
                - batch_size(t): the record of batch size
        """
        def dl_dt(x, t):
          phi = self.phi_interpolator(t)
          var = self.var_interpolator(t)
          topology = self.loss_fn_wrapper.g_value(x, var)
          return - phi * topology

        return odeint(dl_dt, self.initial_loss, self.time_steps)
