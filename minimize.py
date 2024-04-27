from scipy.optimize import minimize
from scipy.optimize import approx_fprime
import numpy as np
def minimize_func(func, x0, hfunc):

    
    return minimize(func, x0, method='SLSQP',constraints={'type':'eq', 'fun': hfunc}, options={'maxiter':400, 'ftol':1e-10})