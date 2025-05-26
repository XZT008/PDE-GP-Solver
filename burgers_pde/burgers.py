import pickle

import torch
from jax.lib import xla_bridge
from numpy import random
import numpy as np
from kernels import *
from get_kernel_matrix import *
import random
import optax
import jax
import jax.numpy as jnp
import tensorly as tl
#from jax.config import config
from time import time
import math

#config.update("jax_enable_x64", True)
print("Jax on", xla_bridge.get_backend().platform)
tl.set_backend('jax')

global nu

@jit
def u(x1, x2):
    return -jnp.sin(jnp.pi * x2) * (x1 == 0) + 0 * (x2 == 0)


[Gauss_pts, weights] = np.polynomial.hermite.hermgauss(80)


def u_truth(x1, x2):
    temp = x2 - jnp.sqrt(4 * nu * x1) * Gauss_pts
    val1 = weights * jnp.sin(jnp.pi * temp) * jnp.exp(-jnp.cos(jnp.pi * temp) / (2 * jnp.pi * nu))
    val2 = weights * jnp.exp(-jnp.cos(jnp.pi * temp) / (2 * jnp.pi * nu))
    return -jnp.sum(val1) / jnp.sum(val2)


class GP_Solver:

    def __init__(self, X_test, Y_test, sensors_test, num_s, num_t, jitter):
        self.jitter = jitter   # tunable 6
        self.num_s = num_s # num. of collocation points for each dim.
        self.num_t = num_t
        self.num = [self.num_s, self.num_t]
        self.d = 2
        self.X_test = sensors_test    # locations for test data
        self.Y_test = Y_test.reshape(-1)
        self.lb = X_test.min(axis=0)
        self.ub = X_test.max(axis=0)
        self.X_col = [np.linspace(self.lb[i], self.ub[i], self.num[i]) for i in range(self.lb.shape[0])]
        self.N_col = (self.num[0])*(self.num[1])

        self.r1 = vmap(u)(np.zeros(self.num[0]), np.linspace(self.lb[0], self.ub[0], self.num[0]))    # four boundaries
        self.r2 = vmap(u)(
            np.linspace(self.lb[1], self.ub[1], self.num[1]),
            np.zeros(self.num[1]),
        )
        self.r3 = jnp.zeros(self.num[1])

        #self.r1 = vmap(u)(np.zeros(self.num[0]), np.linspace(self.lb[0], self.ub[0], self.num[0]))  # four boundaries
        #self.r2 = jnp.zeros(self.num[1])
        #self.r3 = jnp.zeros(self.num[1])
        self.Kernel = Kernel(self.jitter, Matern(jnp.inf))    # nu is tunable
        self.ls = None
        self.fix_ls = False
        self.K_list = None
        self.cho_list = None
        self.K_inv_list = None
        self.derivative_cov_list = None
        self.deri_cov_times_K_inv_list = None

    def init_K(self):
        ls = self.ls
        self.K_list = [self.Kernel.get_kernel_matrix(self.X_col[i], ls[i]) for i in range(2)]
        self.cho_list = [jnp.linalg.cholesky(self.K_list[i]) for i in range(2)]
        self.K_inv_list = [jnp.linalg.inv(self.K_list[i]) for i in range(2)]
        self.derivative_cov_list = self.Kernel.get_derivative_cov([self.X_col[0].T, self.X_col[1].T], ls)
        self.deri_cov_times_K_inv_list = [[jnp.matmul(self.derivative_cov_list[i][j], self.K_inv_list[j]) for j in
                                           range(len(self.derivative_cov_list[0]))]
                                          for i in range(len(self.derivative_cov_list))]
    @partial(jit, static_argnums=(0, ))
    def loss(self, params, coef_eq, tau, v):
        if self.fix_ls:
            ls = self.ls
        else:
            ls = jnp.exp(params['log_ls'])
        mu = params['mu'].sum(axis=-1)

        mu = tl.tenalg.multi_mode_dot(mu, self.cho_list)
        deri_list = [tl.tenalg.multi_mode_dot(mu, self.deri_cov_times_K_inv_list[i]) for i in range(len(self.deri_cov_times_K_inv_list))]

        u_dx1 = deri_list[1].reshape(1, -1)    # get derivatives
        u = deri_list[0].reshape(1, -1)
        u_dx2 = deri_list[2].reshape(1, -1)
        u_ddx1 = deri_list[3].reshape(1, -1)
        u = u.reshape(self.num[0], self.num[1])
        bound1 = u[:, 0].reshape(-1)
        bound2 = u[0, :].reshape(-1)
        bound3 = u[-1, :].reshape(-1)
        mu_K_inv_mu = (params['mu'].sum(axis=-1)**2).sum()

        log_ll_eq = coef_eq * v * jnp.exp(params['log_v']) * jnp.sum(jnp.square(u_dx2.reshape(-1, 1) + u.reshape(-1, 1) * u_dx1.reshape(-1, 1) - nu * u_ddx1.reshape(-1, 1)))
        log_ll_boundaries = tau * 0.5 * jnp.exp(params['log_tau']) * jnp.sum(jnp.square(bound1.reshape(-1) - self.r1.reshape(-1))) + tau * 0.5 * jnp.exp(params['log_tau']) * jnp.sum(
            jnp.square(bound2.reshape(-1) - self.r2.reshape(-1))) + tau * 0.5 * jnp.exp(params['log_tau']) * jnp.sum(jnp.square(bound3.reshape(-1) - self.r3.reshape(-1)))
        #elbo = -0.5 * mu_K_inv_mu + v * 0.5 * params['log_v'] + tau * 0.5 * params['log_tau'] - log_ll_eq - log_ll_boundaries
        elbo = -0.5 * mu_K_inv_mu - log_ll_eq - log_ll_boundaries
        #  elbo = -0.5 * mu_K_inv_mu + v * 0.5 * self.N_col * params['log_v'] + tau * 0.5 * (2*self.num[1]+self.num[0]) * 3 * params['log_tau'] - log_ll_eq
        return -elbo.sum()

    @partial(jit, static_argnums=(0, ))
    def pred(self, params_f):
        if self.fix_ls:
            ls = self.ls
        else:
            ls = jnp.exp(params_f['log_ls'])
        X_col = self.X_col
        mu = params_f['mu'].sum(axis=-1)
        cov_te = [self.Kernel.get_cov(self.X_test[i], X_col[i], ls[i]) for i in range(2)]

        mu = tl.tenalg.multi_mode_dot(mu, self.cho_list)

        K_inv_mu = tl.tenalg.multi_mode_dot(mu, self.K_inv_list)
        pred = tl.tenalg.multi_mode_dot(K_inv_mu, cov_te).reshape(-1)
        return jnp.array(pred.reshape(-1))

    @partial(jit, static_argnums=(0, 1))
    def step(self, optimizer, params, opt_state, coef_eq, tau, v):
        loss, d_params = jax.value_and_grad(self.loss)(params, coef_eq, tau, v)
        updates, opt_state = optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def train(self, fix_ls=False, ls=None, epochs=100000):
        # self.ls = np.array([0.03, 0.06])
        if fix_ls:
            self.ls = np.array(ls)
            self.fix_ls = True
            params = {
                'mu': np.zeros((self.num[0], self.num[1], 1)),
                'log_tau': 0.0,
                'log_v': 0.0,
            }
        else:
            params = {
                # 'mu': 0.001 * np.random.randn(self.num, self.num, 1),
                'mu': np.zeros((self.num[0], self.num[1], 1)),
                'log_ls': np.array([-3.0, -3.0]),
                'log_tau': 0.0,
                'log_v': 0.0,
            }
        optimizer1 = optax.adam(1e-3)
        opt_state = optimizer1.init(params)
        optimizer = optimizer1
        start_EQD = 800    #2000
        coef_eq = 0
        min_err = 1.0
        tau = 1e22      # 20
        v = 1e15        # 15
        print(f"S: {self.num_s}, T: {self.num_t}")
        start_time = time()
        self.init_K()
        for i in range(epochs):
            params, opt_state, _ = self.step(optimizer, params, opt_state, coef_eq, tau, v)
            if i + 1 == start_EQD:
                coef_eq = 1.0
                optimizer2 = optax.adam(1e-4) # tunable learning rate
                opt_state = optimizer2.init(params)
                optimizer = optimizer2
            if (i + 1) % 1000 == 0:
                end_time = time()
                pred = self.pred(params)
                MSE = (((self.Y_test.reshape(-1) - pred.reshape(-1))**2).mean())**0.5
                if MSE < min_err:
                    min_err = MSE

                print("Iter ", i, "MSE ", MSE, ' min ', min_err, "time", end_time-start_time)
                start_time = time()
        return min_err


configs = [
    (0.02, 42, 14, 1e-8, 0.05, 0.3, 100000),            # checked
    (0.02, 60, 20, 1e-8, 0.05, 0.3, 100000),            # checked
    (0.02, 84, 28, 1e-9, 0.04, 0.2, 100000),            # checked
    (0.02, 70, 70, 1e-9, 0.05, 0.2, 100000),            # checked
    (0.001, 42, 14, 1e-8, 0.05, 0.3, 100000),           # checked
    (0.001, 60, 20, 1e-8, 0.05, 0.3, 100000),           # checked
    (0.001, 84, 28, 1e-9, 0.04, 0.2, 100000),           # checked
    (0.001, 120, 40, 1e-9, 0.025, 0.1, 300000),         # checked
    (0.001, 360, 120, 1e-9, 0.0084, 0.0336, 100000),    # checked
    (0.001, 450, 150, 1e-9, 0.0076, 0.0304, 50000),     # checked
    (0.001, 540, 180, 1e-9, 0.0061, 0.024, 50000),      # checked
    (0.001, 600, 200, 1e-9, 0.005, 0.022, 50000),       # checked
]


def run(ns, nt, Jitter, ls_s, ls_t, epochs):
    X = np.linspace(-1, 1, 60)
    T = np.linspace(0, 1.0, 60)

    xx, tt = np.meshgrid(X, T)
    xx = xx.T
    tt = tt.T

    X_test = np.concatenate((xx.reshape(-1, 1), tt.reshape(-1, 1)), axis=1)
    Y_test = vmap(u_truth)(X_test[:, 1], X_test[:, 0])

    sensors_test = [X, T]
    random.seed(0)
    np.random.seed(0)
    solver = GP_Solver(X_test, Y_test, sensors_test, ns, nt, jitter=Jitter)
    min_err = solver.train(fix_ls=True, ls=[ls_s, ls_t], epochs=epochs)
    print(f"Final min mse: {min_err}")


if __name__ == "__main__":
    Nu, ns, nt, Jitter, ls_s, ls_t, epochs = configs[0]
    nu = Nu
    run(ns, nt, Jitter, ls_s, ls_t, epochs)

