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
from Cole_Hopf_for_Eikonal import solve_Eikonal
import pickle

#config.update("jax_enable_x64", True)
print("Jax on", xla_bridge.get_backend().platform)
tl.set_backend('jax')


global epsilon
class GP_Solver:

    def __init__(self, X_test, Y_test, sensors_test, num):
        self.jitter = 5e-6    # tunable
        self.num = num    # num. of collocation points for each dim.
        self.d = 2
        self.X_test = sensors_test    # locations for test data
        self.Y_test = Y_test.reshape(-1)
        self.lb = jnp.array([0, 0])
        self.ub = jnp.array([1, 1])
        self.X_col = np.array([np.linspace(self.lb[i], self.ub[i], self.num) for i in range(self.lb.shape[0])])
        self.N_col = (self.num)**self.d
        self.r1 = np.zeros(self.num)    # four boundaries
        self.r2 = np.zeros(self.num)
        self.r3 = np.zeros(self.num)
        self.r4 = np.zeros(self.num)
        self.Kernel = Kernel(self.jitter, Matern(jnp.inf))
        self.ls = None

    def pred(self, params):
        # ls = jnp.exp(params['log_ls'])
        ls = self.ls   # fix length scale
        X_col = self.X_col
        mu = params['mu']
        cov_te = [self.Kernel.get_cov(self.X_test[i], X_col[i], ls[i]) for i in range(self.d)]
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(X_col[i], ls[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        K_inv_mu = tl.tenalg.multi_mode_dot(mu, self.K_inv_list)
        pred = tl.tenalg.multi_mode_dot(K_inv_mu, cov_te).reshape(-1)
        return jnp.array(pred.reshape(-1))

    @partial(jit, static_argnums=(0, ))
    def loss(self, params, coef_eq, tau, v):
        # ls = jnp.exp(params['log_ls'])
        ls = self.ls    # fix length scale
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i], ls[i]) for i in range(self.X_col.shape[0])])
        self.cho_list = jnp.linalg.cholesky(self.K_list)
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        self.cho_K_inv_list = jnp.linalg.cholesky(self.K_inv_list)
        self.cho_K_inv_list = [self.cho_K_inv_list[i].T for i in range(len(self.cho_K_inv_list))]
        mu = params['mu']
        self.derivative_cov_list = self.Kernel.get_derivative_cov(self.X_col.T, ls)
        self.deri_cov_times_K_inv_list = [[jnp.matmul(self.derivative_cov_list[i][j], self.K_inv_list[j]) for j in range(len(self.derivative_cov_list[0]))]
                                          for i in range(len(self.derivative_cov_list))]
        deri_list = [tl.tenalg.multi_mode_dot(mu, self.deri_cov_times_K_inv_list[i]) for i in range(len(self.deri_cov_times_K_inv_list))]
        u_dx1 = deri_list[1].reshape(1, -1)
        u = deri_list[0].reshape(1, -1)
        u_dx2 = deri_list[2].reshape(1, -1)
        u_ddx1 = deri_list[3].reshape(1, -1)
        u_ddx2 = deri_list[4].reshape(1, -1)
        u = u.reshape(self.num, self.num)
        bound1 = u[:, 0].reshape(-1)
        bound4 = u[:, -1].reshape(-1)
        bound2 = u[0, :].reshape(-1)
        bound3 = u[-1, :].reshape(-1)
        mu_K_inv_mu = ((tl.tenalg.multi_mode_dot(mu, self.cho_K_inv_list))**2).sum()
        # equation likelihood, the source is constant 1
        log_ll_eq = coef_eq * v * jnp.sum(jnp.square(u_dx1**2 + u_dx2**2 - 1 - epsilon * u_ddx1 - epsilon * u_ddx2))
        # boundary likelihood
        log_ll_boundaries = tau * jnp.sum(jnp.square(bound1.reshape(-1) - self.r1.reshape(-1))) + tau * jnp.sum(jnp.square(bound2.reshape(-1) - self.r2.reshape(-1))) + tau * jnp.sum(
            jnp.square(bound3.reshape(-1) - self.r3.reshape(-1))) + tau * jnp.sum(jnp.square(bound4.reshape(-1) - self.r4.reshape(-1)))
        elbo = -0.5 * mu_K_inv_mu - log_ll_eq - log_ll_boundaries

        return -elbo.sum()

    @partial(jit, static_argnums=(0, 1))
    def step(self, optimizer, params, opt_state, coef_eq, tau, v):
        loss, d_params = jax.value_and_grad(self.loss)(params, coef_eq, tau, v)
        updates, opt_state = optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def train(self):
        params = {
            'mu': np.zeros((self.num, self.num)),
        }
        #self.ls = jnp.array([0.18])
        self.ls = jnp.array([0.18])
        optimizer1 = optax.adam(1e-3)
        opt_state = optimizer1.init(params)
        optimizer = optimizer1
        start = 800
        coef_eq = 0
        min_err = 1.0
        tau = 1e36    # tunable
        v = 1e30  # tunable
        for i in range(50000):
            params, opt_state, _ = self.step(optimizer, params, opt_state, coef_eq, tau, v)
            if i + 1 == start:
                coef_eq = 1.0
                optimizer2 = optax.adam(1e-4)
                opt_state = optimizer2.init(params)
                optimizer = optimizer
            if (i + 1) % 1000 == 0:
                pred = self.pred(params)
                MSE = (((self.Y_test.reshape(-1) - pred.reshape(-1))**2).mean())**0.5
                if MSE < min_err:
                    min_err = MSE
                print("Iter ", i, "MSE ", MSE, ' min ', min_err)
        return min_err


def run(num):
    with open(f'truth_{int(100*epsilon)}.pickle', 'rb') as handle:
        tmp = pickle.load(handle)
    X_test_0 = tmp["X_test"]
    Y_test_0 = tmp["Y_test"]
    Y_test_0 = Y_test_0.T
    X_test_0 = X_test_0.reshape(1998, 1998, 2)
    X_test_0 = X_test_0[::16, :]
    X_test_0 = X_test_0[:, ::16]
    sensors_test = X_test_0[0, :, 1]
    sensors_test = [sensors_test, sensors_test]
    Y_test_0 = Y_test_0[::16, :]
    Y_test_0 = Y_test_0[:, ::16]


    random.seed(0)
    np.random.seed(0)

    solver = GP_Solver(X_test_0, Y_test_0, sensors_test, num)
    solver.train()


if __name__ == "__main__":
    epsilon = 0.1
    run(num=18)         # 18, 25, 35, 49

