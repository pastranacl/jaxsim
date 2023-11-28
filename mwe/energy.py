import numpy as np
from jax import grad, vmap
import jax.numpy as jnp

from jax import jit

f32 = jnp.float32
f64 = jnp.float64


TWO_POW_SIX = 2**(1/6)

"""
def force_fn(total_energy):
  return grad(lambda r_coords, **kwargs: -total_energy(r_coords, **kwargs) )
"""


def confinement_cylinder(r_coords,
                          R_conf: float,
                          k_conf: float):
    rdist = jnp.linalg.norm(r_coords[:,1:2])
    
    er = jnp.where(rdist>R_conf, 
                   0.5*k_conf*(rdist - R_conf)**2,
                   0)
    
    return jnp.sum(er)/2


def weeks_chandler_andersen(r_coords, 
                            sigma: float, 
                            eps: float):
    """
        Naive implementation WCA repulsion potential
    """
    rcut = sigma*TWO_POW_SIX
    pair_dist =  jnp.linalg.norm(r_coords[:,None] - r_coords, axis=2)
    wca = jnp.where( jnp.logical_and(pair_dist>0, pair_dist < rcut), 
                    4*eps*(sigma/pair_dist)**12, 
                    0)
    return jnp.sum(wca)/2








