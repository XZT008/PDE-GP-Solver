from jax.lib import xla_bridge
from numpy import random
import numpy as np
from get_kernel_matrix import *
import random
import optax
import jax
import jax.numpy as jnp
from jax import hessian
import tensorly as tl
#from jax.config import config
import pickle
import matplotlib.pyplot as plt

#config.update("jax_enable_x64", True)
print("Jax on", xla_bridge.get_backend().platform)
tl.set_backend('jax')


alpha = 1.0
m = 3.0
global coeff

@jit
def u(x1, x2):
    # return jnp.sin(jnp.pi * x1) * jnp.sin(jnp.pi * x2) + 2 * jnp.sin(4 * jnp.pi * x1) * jnp.sin(4 * jnp.pi * x2)
    return jnp.sin(coeff*2.0*jnp.pi*x1)*jnp.cos(coeff*2.0*jnp.pi*x2) + jnp.sin(2*jnp.pi*x1)*jnp.cos(2*jnp.pi*x2)

# source term
@jit
def f(x1, x2):
    return grad(grad(u, 0), 0)(x1, x2) + grad(grad(u, 1), 1)(x1, x2) + alpha * (u(x1, x2) ** m - u(x1, x2))

class GP_Solver:

    def __init__(self, X_test, Y_test, sensors_test, num=35, jitter=2.5e-5, nu_idx=3):
        self.jitter = jitter           # tunable, added
        self.num = num              # num. of collocation points for each dim., added
        self.d = 2
        self.X_test = sensors_test    # locations for test data
        self.Y_test = Y_test.reshape(-1)
        self.lb = X_test.min(axis=0)
        self.ub = X_test.max(axis=0)
        self.X_col = np.array([np.linspace(self.lb[i], self.ub[i], self.num) for i in range(self.lb.shape[0])])
        # self.X_col = np.array([np.concatenate([np.linspace(0, 0.1, 6), np.linspace(0.11, 0.89, 23), np.linspace(0.9, 1.0, 6)]) for i in range(self.lb.shape[0])])

        self.N_col = (self.num)**self.d
        self.r1 = vmap(u)(self.X_col[0], np.zeros(self.num)).reshape(-1)  # four boundaries
        self.r2 = vmap(u)(np.zeros(self.num), self.X_col[0]).reshape(-1)
        self.r3 = vmap(u)(np.ones(self.num), self.X_col[0]).reshape(-1)
        self.r4 = vmap(u)(self.X_col[0], np.ones(self.num)).reshape(-1)
        xx, tt = np.meshgrid(self.X_col[0], self.X_col[1])
        xx = xx.T
        tt = tt.T
        X_col = np.concatenate((xx.reshape(-1, 1), tt.reshape(-1, 1)), axis=1)
        self.f = vmap(f)(X_col[:, 0], X_col[:, 1]).reshape(-1)    # source term

        nu_list = [0.5, 1.5, 2.5, jnp.inf, 3.5, 4.5]
        self.Kernel = Kernel(self.jitter, Matern(nu_list[nu_idx]))    # nu is tunable, added

        self.ls_transform = None
        self.ls = None

    def Gram_matrix(self):
        ls = self.ls
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(self.X_col[i], ls[i]) for i in range(self.X_col.shape[0])])
        self.cho_list = jnp.linalg.cholesky(self.K_list)
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        self.cho_K_inv_list = jnp.linalg.cholesky(self.K_inv_list)
        self.cho_K_inv_list = [self.cho_K_inv_list[i].T for i in range(len(self.cho_K_inv_list))]
        self.derivative_cov_list = self.Kernel.get_derivative_cov(self.X_col.T, ls)
        self.deri_cov_times_K_inv_list = [[jnp.matmul(self.derivative_cov_list[i][j], self.K_inv_list[j]) for j in range(len(self.derivative_cov_list[0]))]
                                          for i in range(len(self.derivative_cov_list))]

    @partial(jit, static_argnums=(0, ))
    def loss(self, mu, coef_eq, tau, v):
        """
        if self.ls_transform == "exp":
            ls = jnp.exp(params['unconstrained_ls'])
        elif self.ls_transform == "softplus":
            ls = jnp.log(1 + jnp.exp(params['unconstrained_ls']))
        else:
            raise NotImplementedError
        """
        # ls = self.ls
        if self.opt_method == "GN":
            mu = mu.reshape(self.num, self.num)
        deri_list = [tl.tenalg.multi_mode_dot(mu, self.deri_cov_times_K_inv_list[i]) for i in range(len(self.deri_cov_times_K_inv_list))]
        u_ddx1 = deri_list[1].reshape(1, -1)    # get derivatives
        u = deri_list[0].reshape(1, -1)
        u_ddx2 = deri_list[2].reshape(1, -1)
        u = u.reshape(self.num, self.num)

        bound1 = u[:, 0].reshape(-1)
        bound4 = u[:, -1].reshape(-1)
        bound2 = u[0, :].reshape(-1)
        bound3 = u[-1, :].reshape(-1)
        mu_K_inv_mu = ((tl.tenalg.multi_mode_dot(mu, self.cho_K_inv_list))**2).sum()
        log_ll_eq = coef_eq * v * jnp.sum(jnp.square(u_ddx1.reshape(-1) + u_ddx2.reshape(-1) + u.reshape(-1)**3 - u.reshape(-1) - self.f.reshape(-1)))
        log_ll_boundaries = tau * jnp.sum(jnp.square(bound1.reshape(-1) - self.r1.reshape(-1))) + tau * jnp.sum(jnp.square(bound2.reshape(-1) - self.r2.reshape(-1))) + tau * jnp.sum(
            jnp.square(bound3.reshape(-1) - self.r3.reshape(-1))) + tau * jnp.sum(jnp.square(bound4.reshape(-1) - self.r4.reshape(-1)))
        elbo = 0.5*mu_K_inv_mu + log_ll_eq + log_ll_boundaries
        return elbo.sum()

    @partial(jit, static_argnums=(0,))
    def GN_loss(self, mu, mu_old, coef_eq, tau, v):
        # linearize tau for hessian matrix
        mu = mu.reshape(self.num, self.num)
        mu_old = mu_old.reshape(self.num, self.num)
        deri_list = [tl.tenalg.multi_mode_dot(mu, self.deri_cov_times_K_inv_list[i]) for i in
                     range(len(self.deri_cov_times_K_inv_list))]
        u_ddx1 = deri_list[1].reshape(1, -1)  # get derivatives
        u = deri_list[0].reshape(1, -1)
        u_ddx2 = deri_list[2].reshape(1, -1)
        u = u.reshape(self.num, self.num)

        deri_list_old = [tl.tenalg.multi_mode_dot(mu_old, self.deri_cov_times_K_inv_list[i]) for i in
                     range(len(self.deri_cov_times_K_inv_list))]
        u_old = deri_list_old[0].reshape(1, -1)
        u_old = u_old.reshape(self.num, self.num)

        bound1 = u[:, 0].reshape(-1)
        bound4 = u[:, -1].reshape(-1)
        bound2 = u[0, :].reshape(-1)
        bound3 = u[-1, :].reshape(-1)
        mu_K_inv_mu = ((tl.tenalg.multi_mode_dot(mu, self.cho_K_inv_list)) ** 2).sum()

         # p = mu.reshape(1, -1).reshape(self.num, self.num) - mu_old.reshape(1, -1).reshape(self.num, self.num)
        p = mu - mu_old

        log_ll_eq = coef_eq * v * jnp.sum(
            jnp.square(u_ddx1.reshape(-1) + u_ddx2.reshape(-1) + - u.reshape(-1) + u_old.reshape(-1) ** 3 + (3.0*u_old.reshape(-1)**2)*p.reshape(-1) - self.f.reshape(-1)))

        log_ll_boundaries = tau * jnp.sum(jnp.square(bound1.reshape(-1) - self.r1.reshape(-1))) + tau * jnp.sum(
            jnp.square(bound2.reshape(-1) - self.r2.reshape(-1))) + tau * jnp.sum(
            jnp.square(bound3.reshape(-1) - self.r3.reshape(-1))) + tau * jnp.sum(
            jnp.square(bound4.reshape(-1) - self.r4.reshape(-1)))
        elbo = 0.5*mu_K_inv_mu + log_ll_eq + log_ll_boundaries
        return elbo.sum()

    @partial(jit, static_argnums=(0,))
    def Hessian_GN(self, mu, mu_old, coef_eq, tau, v):
        return hessian(self.GN_loss)(mu, mu_old, coef_eq, tau, v)

    @partial(jit, static_argnums=(0,))
    def grad_loss(self, mu, coef_eq, tau, v):
        return grad(self.loss)(mu, coef_eq, tau, v)

    @partial(jit, static_argnums=(0, ))
    def pred(self, mu):
        """
        if self.ls_transform == "exp":
            ls = jnp.exp(params['unconstrained_ls'])
        elif self.ls_transform == "softplus":
            ls = jnp.log(1+jnp.exp(params['unconstrained_ls']))
        else:
            raise NotImplementedError
        """
        ls = self.ls
        X_col = self.X_col
        cov_te = [self.Kernel.get_cov(self.X_test[i], X_col[i], ls[i]) for i in range(self.d)]
        self.K_list = jnp.array([self.Kernel.get_kernel_matrix(X_col[i], ls[i]) for i in range(self.X_col.shape[0])])
        self.K_inv_list = jnp.linalg.inv(self.K_list)
        K_inv_mu = tl.tenalg.multi_mode_dot(mu, self.K_inv_list)
        pred = tl.tenalg.multi_mode_dot(K_inv_mu, cov_te).reshape(-1)
        return jnp.array(pred.reshape(-1))

    @partial(jit, static_argnums=(0, 1))
    def step(self, optimizer, params, opt_state, co, tau, v):
        loss, d_params = jax.value_and_grad(self.loss)(params, co, tau, v)
        updates, opt_state = optimizer.update(d_params, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def train(self, tau=5e6, v=8.0, mu_init="rdm", mu_var=0.1, ls_init=0.5, method="GD", epochs=250000):
        self.opt_method = method

        if mu_init == "rdm":
            mu = np.random.randn(self.num, self.num) * mu_var
        elif mu_init == "ones":
            mu = np.ones((self.num, self.num))
        elif mu_init == "zeros":
            mu = np.zeros((self.num, self.num))
        else:
            raise NotImplementedError

        self.ls = np.array([ls_init]*self.d)
        coef_eq = 1.0    # first fit the boundaries condition only
        min_err = 1.0

        self.Gram_matrix()

        MSE_list = []
        if method == "GD":
            optimizer = optax.adam(1e-4)
            opt_state = optimizer.init(mu)
            for i in range(epochs):
                mu, opt_state, _ = self.step(optimizer, mu, opt_state, coef_eq, tau, v)
                if (i + 1) % 5000 == 0:
                    pred = self.pred(mu)
                    MSE = (((self.Y_test.reshape(-1) - pred.reshape(-1))**2).mean())**0.5
                    if MSE < min_err:
                        min_err = MSE
                    print("Iter ", i, "MSE ", MSE, ' min ', min_err)
                    MSE_list.append(MSE)
        elif method == "GN":
            step_size = 1.0
            coef_eq = 1.0
            self.opt_method = "GN"
            for i in range(1, 200):
                direction = jnp.linalg.solve(self.Hessian_GN(mu.reshape(-1),mu.reshape(-1), coef_eq, tau, v),
                                             self.grad_loss(mu.reshape(-1), coef_eq, tau, v))

                mu = mu - step_size * direction.reshape(self.num, self.num)
                pred = self.pred(mu)
                MSE = (((self.Y_test.reshape(-1) - pred.reshape(-1)) ** 2).mean()) ** 0.5
                if MSE < min_err:
                    min_err = MSE

                print("Iter ", i, "MSE ", MSE, ' min ', min_err)
                MSE_list.append(MSE)
        else:
            raise NotImplementedError
        return min_err


configs = [
    (15.0, 25, 5e-6, 0.01, 10000),
    (15.0, 35, 5e-6, 0.04, 10000),
    (15.0, 49, 5e-6, 0.04, 100000),
    (15.0, 70, 5e-6, 0.04, 250000),

    (15.0, 80, 5e-6, 0.03, 100000),
    (15.0, 90, 5e-6, 0.03, 300000),
    (15.0, 150, 5e-6, 0.03, 500000),
    (15.0, 200, 5e-6, 0.03, 500000),

    (20.0, 25, 5e-6, 0.005, 10000),
    (20.0, 35, 5e-6, 0.005, 10000),
    (20.0, 49, 5e-6, 0.04, 100000),
    (20.0, 70, 5e-6, 0.03, 250000),

    (20.0, 80, 5e-6, 0.03, 200000),
    (20.0, 90, 5e-6, 0.03, 300000),
    (20.0, 150, 5e-6, 0.03, 500000),
    (20.0, 200, 5e-6, 0.03, 500000),
]


def run(num, Jitter, Ls, epochs):
    # mesh for test data
    X = np.linspace(0, 1, 60)
    T = np.linspace(0, 1, 60)
    xx, tt = np.meshgrid(X, T)
    xx = xx.T
    tt = tt.T

    X_test = np.concatenate((xx.reshape(-1, 1), tt.reshape(-1, 1)), axis=1)
    Y_test = vmap(u)(X_test[:, 0], X_test[:, 1]).reshape(60, 60)
    sensors_test = [X, T]

    solver = GP_Solver(X_test, Y_test, sensors_test, nu_idx=3, jitter=Jitter, num=num)
    min_err = solver.train(v=1e4, tau=1e12, mu_var=0.1, ls_init=Ls, mu_init="zeros", method="GD", epochs=epochs)
    print(f"Final min mse: {min_err}")


if __name__ == "__main__":
    Coeff, num, Jitter, Ls, epochs = configs[0]
    coeff = Coeff
    random.seed(0)
    np.random.seed(0)
    run(num, Jitter, Ls, epochs)



