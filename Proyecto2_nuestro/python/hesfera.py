import numpy as np

def hesfera(x):
    n = len(x)
    num_puntos = n//3
    h = np.zeros(num_puntos)
    for j in range(num_puntos):
        uj = x[3*j:3*(j+1)]
        h[j] = np.dot(uj, uj) - 1
    return h