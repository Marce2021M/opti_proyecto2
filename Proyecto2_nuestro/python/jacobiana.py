import numpy as np

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
