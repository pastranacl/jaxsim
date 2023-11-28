bdpars = {
    "kBT": 4.1,                             # Energy room temperature [pN nm]
    "dt": 0.01                             # Time step (s)
}

partpars = {
    "eps": 4*bdpars['kBT'],                    # Repulsion energy particles [pN nm]
    "sigma": 10,                               # Radius of the particle [nm]
    "D": 20                                    # Efective diffusion coefficient [XXX]
}

tubepars = {
    "R_conf": 50,                              # Confinement radius of the tube [nm]
    "k_conf": 2
}