#%% gLV model
import numpy as np
from scipy.integrate import solve_ivp
import jax 
import matplotlib.pyplot as plt
@jax.jit
def gLV(t,x,args):
    r, A = args
   
    dxdt = x * (r + A@x)
    return dxdt

def simulate_gLV(r, A, x0, t):
    args = (r, A)
    sol = solve_ivp(gLV, [t[0], t[-1]], x0, args = (args,), t_eval = t,
                    method = 'BDF', jac = jax.jacfwd(gLV, argnums = 1))
    return sol

if __name__ == '__main__':
    np.random.seed(42)
    r = np.array([1,3,3,4,3])*0.1
    A = np.random.uniform(-1, 1, (5,5))*0.8
    A = A - np.diag(np.diag(A))
    A -= np.eye(5)  

    x0 = np.array([1,1,1,1,1])/10
    t = np.linspace(0, 48, 5)
    sol = simulate_gLV(r, A, x0, t)
    fig, ax = plt.subplots(1,2, figsize = (10,5), dpi = 600)
    t = np.linspace(0, 48, 100)
    sol = simulate_gLV(r, A, x0, t)
    ax[0].plot(sol.t, sol.y.T, label = ['s1', 's2', 's3', 'm1', 'm2'])
    ax[0].legend()
    ax[0].set_xticks( np.linspace(0, 48, 5))
    ax[0].set_yticks( np.linspace(0, 0.4, 5))
    ax[0].set_xlabel('Time [a.u.]')
    ax[0].set_ylabel('Population Growth [a.u.]')
    ax[0].set_title('gLV model simulation')
    
    x0 = np.array([1,1,1,1,1])/10
    t = np.linspace(0, 48, 5)
    sol = simulate_gLV(r, A, x0, t)
   
    for i in range(5):
        solution = sol.y[i] + np.random.normal(0, 0.005, len(sol.y[i])) 
        ax[1].scatter(sol.t, solution, label = f's{i}')
    ax[1].legend()
    t = np.linspace(0, 48, 100)
    sol = simulate_gLV(r, A, x0, t)
    solu = sol.y + np.random.normal(0, 0.005, sol.y.shape)
    ax[1].plot(sol.t, solu.T, alpha = 0.5)
    ax[1].set_xticks( np.linspace(0, 48, 5))
    ax[1].set_yticks( np.linspace(0, 0.4, 5))
    ax[1].set_xlabel('Time [a.u.]')
    ax[1].set_ylabel('Population Growth [a.u.]')
    ax[1].set_title('gLV model simulation with noise')
    plt.show()
