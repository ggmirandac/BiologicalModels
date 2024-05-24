import numpy as np
import jax 
import scipy.integrate as spi   


#%% Model

@jax.jit
def multiple_interactions(t, x, args):
    A, B, C = args
    return x * (A @ x + B @ x @ x + C @ x @ x @ x)

#%% Simulation
def simulate_multiple_interactions(A, B, C, x0, t):
    args = (A, B, C)
    sol = spi.solve_ivp(multiple_interactions, [t[0], t[-1]], x0, args = args, t_eval = t, 
                        method = 'BDF', jac = jax.jacfwd(multiple_interactions, argnums = 1))
    return sol