import numpy as np

from jax import random
import jax.numpy as jnp
from jax import lax
from jax import grad, vmap

import integrator as bd
from energy import confinement_cylinder
from energy import weeks_chandler_andersen
import simparams as simp
import exporter as dump

from tqdm import tqdm

   


if __name__ == '__main__':

    N_PARTS = int(20)
    DEVICE_STEPS = int(2)
    HOST_STEPS = int(10)
    
    key = random.PRNGKey(0) # Initialise Random number generator
    r_coords = random.uniform(key, (N_PARTS, 3), dtype=jnp.float32,
                             minval=0.0,
                             maxval=2*simp.tubepars['R_conf'])
   
    # Define energy functions
    confinement_fn = lambda r_coords: confinement_cylinder(r_coords, simp.tubepars['R_conf'], simp.tubepars['k_conf'])
    repwca_fn = lambda r_coords: weeks_chandler_andersen(r_coords, simp.partpars['sigma'], simp.partpars['eps'])
    energy_fn = lambda r_coords: confinement_fn(r_coords) + repwca_fn(r_coords)
    vmap(energy_fn)

    # Initialise
    bdstep = bd.brownian(energy_fn,
                    simp.bdpars['dt'],
                    simp.bdpars['kBT'],
                    simp.partpars['D'],)

    # MAIN SIMULATION LOOP
    dump.save_xyz(r_coords, N_PARTS, 0)
    for hs in tqdm(range(0, HOST_STEPS)):
        
        # Steps on the device
        """
        r_coords, key = lax.fori_loop(0, DEVICE_STEPS, 
                                        bdstep, (r_coords, key) )
        """
        r_coords, key =  bdstep(hs, (r_coords, key))
        # Save the coordinates
        dump.save_xyz(r_coords, N_PARTS, hs+1)

        