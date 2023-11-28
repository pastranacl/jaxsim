import jax.numpy as jnp
from jax import grad
from jax import random


def brownian(energy_fn,
            dt: float, 
            kBT: float,
            D: float):

    #force_fn = grad(energy_fn)
    force_fn = grad(lambda x: -energy_fn(x))
    
    def bdstep(t, state):

        r_coords, key = state
        
        key, split = random.split(key)
        xi = random.normal(split, r_coords.shape, r_coords.dtype)

        F = force_fn(r_coords)
        dr = F*dt*D/kBT + jnp.sqrt(2*D*dt)*xi

        r_coords += dr

        return (r_coords, key)
    
    return bdstep 

