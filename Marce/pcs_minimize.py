# Integrantes del equipo:

# --------------------------------------------------------------
# Librerias que utiliza nuestro archivo
# --------------------------------------------------------------

from scipy.optimize import minimize
from scipy.optimize import approx_fprime
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
from scipy.optimize import SR1
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

# --------------------------------------------------------------
# Código para resolver el problema de optimización
# --------------------------------------------------------------

# Definimos el número de puntos aleatorios que vamos a organizar
# en la esfera menos 1 porque en la optimización se fija un punto de los np +1
# para tener solución única.
np.random.seed(191654)

num_points_random = 20 # número de puntos que se solicitaron

# Se generan los puntos aleatorios en el espacio para inicializar el método
# de optimización y se normalizan para que estén en la esfera.

initial_points = np.random.randn(3 * num_points_random)
initial_norms = np.linalg.norm(initial_points.reshape(num_points_random, 3), axis=1)
initial_points /= initial_norms[:, np.newaxis].repeat(3, axis=1).flatten()

# Se procede a optimizar la función
start_time = time.time() # Medir tiempo de ejecución
resultado = minimize_func(f_electron, initial_points, h_esfera)
cpu_time = time.time() - start_time

# Calculamos gradiente de f
grad_fObj = approx_fprime(resultado.x, f_electron, 1e-6)
# Calculamos gradiente de h
grad_h = approx_fprime(resultado.x, h_esfera, 1e-6)
# Calculamos multiplicadores de lagrange
lambda_ = np.linalg.solve(np.dot(grad_h, grad_h.T), np.dot(grad_h, -grad_fObj))
# Calculamos el gradiente de la función lagrangiana
grad_lagrangiana = grad_fObj + np.dot(grad_h.T, lambda_)
# Adjuntamos la condición de primer orden de la restricción
grad_lagrangiana = np.concatenate((grad_lagrangiana,h_esfera(resultado.x) ))
# Calculamos la norma del gradiente de la función lagrangiana
norm_grad_lagrangiana = np.linalg.norm(grad_lagrangiana)

# --------------------------------------------------------------
# Presentación de resultados
# --------------------------------------------------------------

# Creación de la tabla

data = {
    "np": [num_points_random +1],
    "CNPO": [norm_grad_lagrangiana],  # Ajusta esto según la interpretación correcta de "CN P O"
    "f(x*)": [resultado.fun],
    "cpu time": [cpu_time],
    "nIter": [resultado.nit],
}

print(pd.DataFrame(data)) # Resultados de la tabla


# Creación de gráfica

# Supongamos que esta es la salida de tu optimización, por ejemplo
points = resultado.x.reshape(-1, 3)

# Crear la figura
fig = go.Figure()

# Añadir los puntos optimizados al gráfico
fig.add_trace(go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='red',                # set color to an array/list of desired values
        colorscale='Viridis',       # choose a colorscale
        opacity=0.8
    )
))

# Añadir la esfera unitaria al gráfico
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.3, showscale=False))

# Actualizar los aspectos del gráfico para igualar los aspectos de los ejes
fig.update_layout(
    scene = dict(
        xaxis=dict(nticks=4, range=[-1,1]),
        yaxis=dict(nticks=4, range=[-1,1]),
        zaxis=dict(nticks=4, range=[-1,1])
    ),
    title_text="Optimización sobre Esfera Unitaria",
    scene_aspectmode='cube'
)

# Mostrar el gráfico
fig.show()
