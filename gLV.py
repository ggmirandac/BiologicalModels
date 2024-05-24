import numpy as np
from scipy.integrate import solve_ivp
import jax 

@jax.jit
#%% Model
def gLV(t,x,args):
    r, A = args
   
    dxdt = x * (r + A@x)
    return dxdt

#%% Simulation

def simulate_gLV(r, A, x0, t):
    args = (r, A)
    sol = solve_ivp(gLV, [t[0], t[-1]], x0, args = args, t_eval = t,
                    method = 'BDF', jac = jax.jacfwd(gLV, argnums = 1))
    return sol