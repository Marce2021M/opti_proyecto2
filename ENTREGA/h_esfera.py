"""
Autores: 
    - Diana Espinosa Ruiz CU:
    - Alfredo Alef Pineda Reyes CU: 191164
    - Marcelino Sanchez Rodriguez CU: 191654
    - Carlos Alberto Delgado Elizondo CU: 181866

"""
import numpy as np

def h_esfera(x):
    n = len(x)
    num_puntos = n//3
    h = np.zeros(num_puntos)
    for j in range(num_puntos):
        uj = x[3*j:3*(j+1)]
        h[j] = np.dot(uj, uj) - 1
    return h