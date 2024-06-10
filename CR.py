#%% Installs and functions
import numpy as np
from scipy.integrate import solve_ivp
import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
# Model

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

    return jnp.concatenate([dsdt, dmdt])

# Simulation

def simulate_CR(t, x0 ,n_s, n_m, C, g, P):
    '''
    Simulates the system from:
    t: time points
    x0: initial conditions
    n_s: number of species
    n_m: number of resources
    C: consumption matrix
    g: growth rate of species
    P: production matrixs
    '''
    args = (n_s, n_m, C, g, P) 
    t = jnp.linspace(0, 30, 2000)
    tspan = [t[0], t[-1]]
    sol = solve_ivp(CR_model, tspan, x0, args=(args,), t_eval=t,
                        method='LSODA', jac = jax.jacfwd(CR_model, argnums=1))

    time = sol.t
    results = sol.y.T
    return time, results

#%%
if __name__ == '__main__':
    t = np.linspace(0, 24, 10)
    x0_s = np.random.dirichlet(np.ones(4)*10)
    x0_m = np.random.dirichlet(np.ones(4)*10)*10
    x0 = jnp.concatenate([x0_s, x0_m])
    n_s = 4
    n_m = 4
    C = np.random.rand(n_s, n_m)
    g = np.random.rand(n_s)
    P = np.random.rand(n_m, n_s)

    #print((x0[:n_s] * (C @ x0_m - g)).shape)
  

    sol = simulate_CR(t, x0, n_s, n_m, C, g ,P)
    plt.plot(sol[0], sol[1])
    print(sol[1].shape)
    # %%
