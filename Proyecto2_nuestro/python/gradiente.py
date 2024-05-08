import numpy as np
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

