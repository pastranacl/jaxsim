import numpy as np

from jax import random
import jax.numpy as jnp
from jax import lax
from jax import grad

from energy import confinement_cylinder
from energy import weeks_chandler_andersen
import simparams as simp

import tqdm

def brownian(energy_fn,
            dt: float, 
            kBT: float,
            D: float):

    force_fn = grad(energy_fn)

    def bdstep(t, state):

        r_coords, key = state
        key, split = random.split(key)

        F = force_fn(r_coords)
        xi = random.normal(split, r_coords.shape, r_coords.dtype)

        dr = F*dt*kBT/D + jnp.sqrt(2*D*dt)*xi
        r_coords += dr

        return (r_coords, key)
    
    return bdstep 




   


if __name__ == '__main__':

    N_PARTS = 1000
    DEVICE_STEPS = 1000
    HOST_STEPS = 10

    r_coords = jnp.zeros((N_PARTS, 3), dtype=np.float32)
    # Initialise Random number generator
    key = random.PRNGKey(0)
    
    # Define energy functions
    confinement_fn = lambda r_coords: confinement_cylinder(r_coords, simp.tubepars['R_conf'], simp.tubepars['k_conf'])
    repwca_fn = lambda r_coords: weeks_chandler_andersen(r_coords, simp.partpars['sigma'], simp.partpars['eps'])

    def energy_fn(r_coords):
        return confinement_fn(r_coords) + repwca_fn(r_coords)
    
    #orces = grad(energy_fn)
    bdstep = brownian(energy_fn,
                    simp.bdpars['dt'],
                    simp.bdpars['kBT'],
                    simp.partpars['D'],)

    for hs in range(0, HOST_STEPS):
       
        #r_coords, key = bdstep( (r_coords,key) )
     
        r_coords, key = lax.fori_loop(0, DEVICE_STEPS, 
                                        bdstep, (r_coords, key) )
        