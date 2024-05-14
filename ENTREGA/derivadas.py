"""
Autores: 
    - Diana Espinosa Ruiz CU: 179164
    - Alfredo Alef Pineda Reyes CU: 191164
    - Marcelino Sanchez Rodriguez CU: 191654
    - Carlos Alberto Delgado Elizondo CU: 181866

"""
import numpy as np

def derivada(f, x0, ep = 1e-7):
    return (f(x0 + ep) - f(x0 - ep))/2*ep
# Calcula el gradiente de f respecto a x evaluado en x0
# Utiliza diferencias centrales
# Regresa este gradiente
def gradiente(fx, x0, ep = 1e-7):
    n = len(x0)
    gf = np.zeros(n)
    x_adelante = x0.copy()
    x_atras = x0.copy()
    for k in range(n):
        x_adelante[k] += ep
        x_atras[k] -= ep
        gf[k] = (fx(x_adelante) - fx(x_atras)) / (2*ep)
        x_adelante[k] -= ep
        x_atras[k] += ep
    return gf

# Calcula la jacobiana de h respecto a x evaluado en x0
# Utiliza diferencias centrales
# Regresa la matriz jacobiana
def jacobiana(hx, x0, ep=1e-7):
    n = len(x0)
    m = len(hx(x0))
    jh = np.zeros((m, n))

    x_adelante = x0.copy()
    x_atras = x0.copy()
    for k in range(n):
        x_adelante[k] += ep
        x_atras[k] -= ep
        jh[:, k] = (hx(x_adelante) - hx(x_atras)) / (2*ep)
        x_adelante[k] -= ep
        x_atras[k] += ep    
    return jh