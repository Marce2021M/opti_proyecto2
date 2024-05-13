# --------------------------------------------------------------
# Librerias que utiliza nuestro archivo
# --------------------------------------------------------------

from scipy.optimize import minimize
from scipy.optimize import approx_fprime
import numpy as np
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
import time
import pandas as pd
import plotly.graph_objects as go

# --------------------------------------------------------------
# Definimos las funciones que vamos a utilizar
# --------------------------------------------------------------

def minimize_func(func, x0, hfunc):
    nonlinear_constraint =NonlinearConstraint(hfunc,0,0, jac='2-point',hess=BFGS())
    return minimize(func, x0, method='SLSQP',constraints={'type':'eq', 'fun': hfunc})

def f_electron(x, puntoInicial = np.array([1,0,0])):
    # x es un vector columna de dimensi ́on 3 ∗ np. (np.array)
    # El punto #, k, xk que quiere colocarse en la esfera,
    # tiene coordenadas (x3∗k, x3k+1, x3∗k+2).
    # Return:
    # f−x valor de la funci ́on a minimizar.
    nproton = len(x)//3 + 1
    x = np.array(x)
    puntoInicial = np.array(puntoInicial)
    x = np.concatenate((puntoInicial, x))
    x = np.reshape(x, (nproton, 3))
    resultado = 0
    for i in range(nproton):
        for j in range(nproton):
            if i < j:
                resultado += 1/np.linalg.norm(x[i]-x[j])
                
    return resultado

def h_esfera(x):
    # versi ́on para usar en programaci ́on cuadr ́atica sucesiva
    # x es un vector columna de dimensi ́on 3 ∗ np.
    # El punto #, k, xk que quiere colocarse en la esfera,
    # tiene coordenadas (x[3 ∗ k], x[3k + 1], x[3 ∗ k + 2]).
    # Return h−x .
    # h−x es el vector de dimensi ́on np que indica
    # si el punto xk est ́a cerca o no de la esfera.
    # h−x[k] = x[3 ∗ k] ∗ ∗(2) + x[3k + 1] ∗ ∗(2) + x[3 ∗ k + 2] ∗ ∗(2) − 1
    #
    x = np.array(x)
    nproton = len(x)//3 
    x = np.reshape(x, (nproton, 3))
    return np.linalg.norm(x, axis = 1)**2 - 1