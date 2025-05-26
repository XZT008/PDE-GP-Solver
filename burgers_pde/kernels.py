import jax.numpy as jnp
from jax import grad, jit
from functools import partial
#from jax.config import config
import jax
import numpy as np
from jax import jacfwd, jacrev
from jax import vmap

#config.update("jax_enable_x64", True)


class Matern(object):

    def __init__(self, nu):
        self.nu = nu

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, ls):
        dist = jnp.abs(x1 - y1) / ls
        if self.nu == 0.5:
            K = jnp.exp(-dist)
        elif self.nu == 1.5:
            K = dist * jnp.sqrt(3)
            K = (1.0 + K) * jnp.exp(-K)
        elif self.nu == 2.5:
            K = dist * jnp.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * jnp.exp(-K)
        elif self.nu == jnp.inf:
            K = jnp.exp(-(dist**2) / 2.0)
        return K

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, ls):
        val = grad(self.kappa, 0)(x1, y1, ls)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, ls):
        val = grad(grad(self.kappa, 0), 0)(x1, y1, ls)
        return val

    @partial(jit, static_argnums=(0, ))
    def DDD_x1_kappa(self, x1, y1, ls):
        val = grad(grad(grad(self.kappa, 0), 0), 0)(x1, y1, ls)
        return val

    @partial(jit, static_argnums=(0, ))
    def DDDD_x1_kappa(self, x1, y1, ls):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 0), 0)(x1, y1, ls)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, y1, ls):
        val = grad(self.kappa, 1)(x1, y1, ls)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, y1, ls):
        val = grad(grad(self.kappa, 1), 1)(x1, y1, ls)
        return val

    @partial(jit, static_argnums=(0, ))
    def DDD_y1_kappa(self, x1, y1, ls):
        val = grad(grad(grad(self.kappa, 1), 1), 1)(x1, y1, ls)
        return val

    @partial(jit, static_argnums=(0, ))
    def DDDD_y1_kappa(self, x1, y1, ls):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 1), 1)(x1, y1, ls)
        return val


class RFF_cosine(object):
    @partial(jit, static_argnums=(0,))
    def kappa(self, x, random_freq, ls, N_freq):
        return jnp.cos(ls * random_freq * x) / jnp.sqrt(N_freq)

    @partial(jit, static_argnums=(0,))
    def D_kappa(self, x, random_freq, ls, N_freq):
        val = grad(self.kappa, 0)(x, random_freq, ls, N_freq)
        return val

    @partial(jit, static_argnums=(0,))
    def DD_kappa(self, x, random_freq, ls, N_freq):
        val = grad(self.kappa, 0)(x, random_freq, ls, N_freq)
        return val


if __name__ == "__main__":
    N_freq = 20
    N = 2400
    kernel = RFF_cosine()
    random_freq = np.random.standard_normal(N_freq).flatten()
    test_x = np.random.standard_normal(N).flatten()

    k_explicit = jnp.cos(1.0 * random_freq.reshape(1, -1) * test_x.reshape(-1, 1)) / jnp.sqrt(N_freq)
    dk_explicit = -1.0 * random_freq.reshape(1, -1) * jnp.sin(1.0 * random_freq.reshape(1, -1) * test_x.reshape(-1, 1)) / jnp.sqrt(N_freq)
    random_freq = jnp.tile(random_freq, (N, 1))
    test_x = jnp.tile(test_x, (N_freq, 1)).T

    k = vmap(kernel.kappa, (0, 0, None, None))(test_x.flatten(), random_freq.flatten(), 1.0, N_freq)
    dk = vmap(kernel.D_kappa, (0, 0, None, None))(test_x.flatten(), random_freq.flatten(), 1.0, N_freq)


    print()
