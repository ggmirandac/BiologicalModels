import numpy as np
import pandas as pd
from CR import CR_model
import scipy.integrate as spi
import jax
import itertools


def gen_leave_one_out(length = 3, sum_row = 1):
    '''
    Generates a matrix with leave one out structure

    sum_row: sum of the elements in the row
    length: amount of elements in the row
    '''
    x0 = np.zeros([length, length]) # is a square matrix
    sum_1_factor = sum_row/(length-1) 
    for i in range(length):
        x0[i] = np.ones(length)*sum_1_factor
        x0[i][i] = 0
    return x0

def gen_mono(length = 3, sum_row = 1):
    '''
    Generates a matrix with mono structure

    sum_row: sum of the elements in the row
    length: amount of elements in the row
    '''
    x0 = np.zeros([length, length]) # is a square matrix
    for i in range(length):
        x0[i,i] = sum_row
    return x0

def gen_random_matrix(length = 3, sum_row = 1, n_samples = 10):
    '''
    Generates a random matrix with the given sum_row

    sum_row: sum of the elements in the row
    length: amount of elements in the row
    '''
    x0 = np.zeros([n_samples, length])
    for i in range(n_samples):
        x0[i] = np.random.dirichlet(np.ones(length)*sum_row)
    return x0


def initial_syn_val(n_samples, args):
    """
    Generate a synthetic data set for the consumer resource model
    n_samples: number of samples not considering mono and leave one out
    args: arguments for the model
    t_eval: time points to evaluate the model
    noise: noise level
    """
    n_s, n_m, C, g, P = args
    mono_s = gen_mono(n_s, 1)
    mono_m = gen_mono(n_m, 3)

    

    leave_one_out_s = gen_leave_one_out(n_s, 1)
    leave_one_out_m = gen_leave_one_out(n_m, 3)
   
    random_s = gen_random_matrix(n_s, 1, n_samples)
    random_m = gen_random_matrix(n_m, 3, n_samples)
    
    data1 = np.vstack([np.concatenate(i) for i in itertools.product(mono_s, mono_m)])
    data2 = np.vstack([np.concatenate(i) for i in itertools.product(mono_s, leave_one_out_m)])
    data3 = np.vstack([np.concatenate(i) for i in itertools.product(mono_s, random_m)])
    data4 = np.vstack([np.concatenate(i) for i in itertools.product(leave_one_out_s, mono_m)])
    data5 = np.vstack([np.concatenate(i) for i in itertools.product(leave_one_out_s, leave_one_out_m)])
    data6 = np.vstack([np.concatenate(i) for i in itertools.product(leave_one_out_s, random_m)])
    data7 = np.vstack([np.concatenate(i) for i in itertools.product(random_s, mono_m)])
    data8 = np.vstack([np.concatenate(i) for i in itertools.product(random_s, leave_one_out_m)])
    data9 = np.vstack([np.concatenate(i) for i in itertools.product(random_s, random_m)])
    data = np.vstack([data1, data2, data3, data4, data5, data6, data7, data8, data9])
    return data

def gen_ground_thruth(n_samples, args, noise = 0.1):
    '''
    Generates dynamical systems ground thruth
    args: arguments for the model
    n_samples: number of samples not considering mono and leave one out
    ----------------
    Notes
    This code is used for the consumer resource model, but can be adapted to other models.
    '''

    initial_samples = initial_syn_val(n_samples, args)
    n_total = initial_samples.shape[0]
    t_span = [0, 24]
    t_eval = np.linspace(t_span[0], t_span[-1], 10)
    n_s, n_m, C, g, P = args

    df_time = pd.DataFrame()


    for i in range(n_total):
        sol = spi.solve_ivp(CR_model, t_span, initial_samples[i], args=(args,), t_eval=t_eval,
                     method='BDF', jac = jax.jacfwd(CR_model, argnums=1))
        df = pd.DataFrame(sol.y.T, columns = ['s1', 's2', 's3', 'm1', 'm2'])
        df.insert(0, 'Time', sol.t)
        df.insert(0, 'Treatments', i)
        df_time = pd.concat([df_time, df])

    return df_time