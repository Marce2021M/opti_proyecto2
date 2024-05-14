"""
Autores: 
    - Diana Espinosa Ruiz CU: 179164
    - Alfredo Alef Pineda Reyes CU: 191164
    - Marcelino Sanchez Rodriguez CU: 191654
    - Carlos Alberto Delgado Elizondo CU: 181866

"""
import numpy as np

def pc(Q, A, c, b):
    # Método directo para resolver el problema
    # min (1/2)*x'Qx + c'*x
    # SA A*x = b
    # Q es una matriz spd
    # A es de mxn con rango(A) = m
    # c vector columna de orden n
    # b vector columna de orden m
    #
    # Out
    # x valor columna de orden n con la sol numerica del problema
    # lambda vector columna de orden que representa el mult de lagrange
    # optimización 25 de agosto

    m = len(b)  # numero de restricciones
    n = len(c)  # numero de variables
    K = np.block([[Q, A.T], [A, np.zeros((m, m))]])
    ld = np.concatenate((-c, b))
    
    # resolver el sistema lineal K*w = ld
    # w = [x; lambda]
    w = np.linalg.solve(K, ld)
    x = w[:n]
    lambd = w[n:n+m]
    
    return x, lambd

