# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:58:45 2024

@Zeferino Paradad

"""

def gradiente(fun,x):
    # Aproximación del vector gradiente de fun:R^n --> R
    # por diferencias hacia adelante
    #
    # In
    # fun. función 
    # x  vector donde se calcula el gradiente.
    # Return
    # grad_x vector gradiente
    #
    # Optimización Numérica
    # ITAM
    # 10 de abril de 2024
    #-----------------------------------------
    import numpy as np
    import copy
    step = 10**(-5)
    fx = fun(x)
    grad_x = np.zeros(len(x))
    #
    for i in range(len(x)):
        x_step = copy.copy(x)
        x_step[i] = x_step[i] + step
        fx_step = fun(x_step)
        grad_x[i] = (fx_step - fx)/step
        # 
    return grad_x
    #-------------------------------------------------------


def jacobiana(fun,x):
    # Matriz jacobiana de h:r^n --> R^m  
    # dos veces continuamente diferenciable en el punto x.
    # Optimización Numérica
    # ITAM
    # 10 de abril de 2024
    #
    # In
    # fun. función 
    # x  vector donde se calcula la jacobiana.
    # Return
    # J matriz jacobiana de mxn calculada por diferencias hacia adelante.
    #--------------------------------------------------------------------
    import numpy as np
    import copy
    fx = fun(x)
    n = len(x)
    m = len(fx)
    J = np.zeros((m,n))
    step = 10**(-5)
    # aproximación a la jacobiana
    for i in range(n):
        x_step = copy.copy(x)
        x_step[i] = x_step[i]+step
        fx_step = fun(x_step)
        J[:,i] = (fx_step - fx)/ step
        #
    return J  
#-------------------------------------------------------------------  

def matriz_rango1(s,y):
    # se genera la matriz nxn de rango 1
    #  M =(s.T)*y
    #  vectores s, y de la misma dimensión
    import numpy as np
    M = np.zeros((len(s), len(s)))
    for i in range(len(s)):
        M[i,:] =s[i]*y
    return M
#----------------------------------------    
        
        
        
        
        
                 
