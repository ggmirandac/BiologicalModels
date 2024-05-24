import numpy as np
from scipy.integrate import solve_ivp
import jax 

#%% Model
@jax.jit
def CR_model(t, x, args):
    """
    Consumer Resouce model
    n_s: number of species
    n_m: number of resources
    C: consumption matrix 
    g: growth rate of species
    P: production matrix
    """
    n_s, n_m, C, g, P = args
    
    s = x[:n_s]
    m = x[n_s:]


    dsdt = s * (C @ m - g)

    dmdt = P @ s - m * (s.T @ C)

    return np.concatenate([dsdt, dmdt])

#%% Simulation

def simulate_CR(C, g, P, x0, t):
    args = (C.shape[0], C.shape[1], C, g, P)
    sol = solve_ivp(CR_model, [t[0], t[-1]], x0, args = args, t_eval = t,
                    method = 'BDF', jac = jax.jacfwd(CR_model, argnums = 1))
    return sol