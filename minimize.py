from scipy.optimize import minimize
from scipy.optimize import approx_fprime
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
from scipy.optimize import SR1

def minimize_func(func, x0, hfunc):
    nonlinear_constraint =NonlinearConstraint(hfunc,0,0, jac='2-point',hess=BFGS())
    #return minimize(func, x0, method='SLSQP',constraints={'type':'eq', 'fun': hfunc}, options={'maxiter':400, 'ftol':1e-10})
    return minimize(func, x0, method='trust-constr',jac = '2-point',hess = SR1(),constraints=nonlinear_constraint, options={'maxiter':1600})