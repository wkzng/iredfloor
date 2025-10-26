from scipy.interpolate import PchipInterpolator  # monotone, no overshoot
from scipy.integrate import odeint
import numpy as np


from src.loss_wrappers import TheoreticalLossWrapper


class LossDynamicsSolver:
    def __init__(self, 
            loss_fn_wrapper:TheoreticalLossWrapper, 
            initial_loss:float, 
            loss_variance: np.ndarray,
            t_values:np.ndarray,
            phi:np.ndarray,
            phi_ema_beta:float=1,
            var_ema_beta:float=1,
            min_eps:float=1e-12
        ):
        """
            loss_fn_wrapper: wrapper for the loss function along with the curvatuve functions
            initial_loss : initial value of the training loss across batche or the entire dataset
            learning_rate: sequence of learning rate (this can be a constant value if the learning rate is fixed)
            phi: sequence of effective rates
            batch_sizes: seauence of batch sizes
        """
        # basic checks
        assert self.t_values.ndim == 1 and self.t_values.size >= 2, "t_values must be 1D with >=2 points"
        assert np.all(np.diff(self.t_values) > 0), "t_values must be strictly increasing"

        self.loss_fn_wrapper = loss_fn_wrapper
        self.initial_loss = initial_loss
        self.t_values = t_values
        self.min_eps = min_eps

        # optional lightweight smoothing (warm-started EMA)
        self.phi = self.ema(phi, beta=phi_ema_beta)
        self.loss_variance = self.ema(loss_variance, beta=var_ema_beta)
        
        # monotone interpolants to avoid spline overshoot on noisy logs
        self._phi = PchipInterpolator(self.t_values, self.phi, extrapolate=True)
        self._var  = PchipInterpolator(self.t_values, self.var,  extrapolate=True)


    def ema(self, x:np.ndarray, beta:float):
        """Apply exponentual moving average on a sequence"""
        y = np.empty_like(x, dtype=float)
        s = 0
        for i, v in enumerate(x):
            s = beta*v + (1-beta)*s
            y[i] = s
        return y

    def solve(self):
        """ Compute numerical solutions to:  dL/dt = - phi(t) * g(L, sigma^2)
                - phi(t): as the effective time varying decay rate
                - g(L,sigma^2): surrogate topology of ||gradL||^2
        """
        def dL_dt(L:float, t:float):
            phi = max(0, float(self._phi(t)))
            var = max(0, float(self._var(t)))
            g_value = max(self.min_eps, float(self.loss_fn_wrapper.g_value(L, variance=var)))
            return - phi * g_value

        # sol = solve_ivp(
        #     fun=dx_dt, 
        #     t_span=(self.t_values[0], self.t_values[-1]), 
        #     y0=[self.initial_loss],
        #     t_eval=self.t_values,
        #     method="RK45",
        #     rtol=1e-6,
        #     atol=1e-9
        # )
        # return sol.y[0]
        return odeint(dL_dt, self.initial_loss, self.t_values)
