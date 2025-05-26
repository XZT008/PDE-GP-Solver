import jax.numpy as jnp
from jax import grad, jit
from functools import partial
#from jax.config import config

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
        elif self.nu == 3.5:
            K = dist * jnp.sqrt(7)
            K = (1.0 + K + (6.0/15.0)*K**2 + (1.0/15.0)*K**3) * jnp.exp(-K)
        elif self.nu == 4.5:
            K = dist * jnp.sqrt(9)
            K = (1.0 + K + (15.0 / 35.0)*K**2 + (10.0/105.0)*K**3 + (1.0/105.0)*K**4) * jnp.exp(-K)
        elif self.nu == jnp.inf:
            K = jnp.exp(-(dist**2) / 2.0)
        else:
            raise NotImplementedError
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
