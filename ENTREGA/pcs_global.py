"""
Autores: 
    - Diana Espinosa Ruiz CU:
    - Alfredo Alef Pineda Reyes CU:
    - Marcelino Sanchez Rodriguez CU: 191654
    - Carlos Alberto Delgado Elizondo CU: 181866

"""
import numpy as np
from derivadas import gradiente, jacobiana, derivada
from pc import pc

# Es el metodo principal del proyecto
# Resuelve de forma iterativa el problema de minimización con 
# programación cuadrática sucesiva, utilizando:
# - Funciones de mérito para encontrar un alfa adecuada
# - Actualización de mínimos cuadrados lineales para multiplicadores de Lagrange
# - Actualización BFGS para la aproximación de la hessiana
def pcs_global(fx, hx, x0):
    n = len(x0)
    m = len(hx(x0))
    tol = 1e-5
    maximo_iteraciones = 300
    iteraciones = 0
    B = np.eye(n)
    multiplicadores_Lagrange = np.zeros(m)

    x = x0
    cnpo = np.concatenate((gradiente_Lagrangeano_x(fx, hx, x, multiplicadores_Lagrange), hx(x)))    
    while iteraciones < maximo_iteraciones and np.linalg.norm(cnpo) >= tol:
        # Resolver el subproblema cuadrático
        pk, multiplicadores_Lagrange = pc(B, jacobiana(hx, x), gradiente(fx, x), -hx(x)) 

        # Encontrar constante para que la direccion p sea direccion de descenso
        # de la funcion de merito
        C_merito = constante_funcion_merito(fx, hx, x, pk, iteraciones)
        
        phi = lambda alpha_x: funcion_merito(fx, hx, x+alpha_x*pk, C_merito)
        # encontrar alfa que cumpla las condiciones de Wolfe
        alpha = busqueda_en_linea(fx, hx, phi, x, pk, C_merito)
        
        # actualizar la aproximacion de la hessiana
        B = BFGS(B, fx, hx, x, C_merito, alpha, pk)
        x += alpha*pk

        # actualizamos los multiplicadores con minimos cuadrados lineales
        jac = jacobiana(hx, x)
        multiplicadores_Lagrange = np.dot(-np.linalg.inv(np.dot(jac , jac.T)), np.dot(jac, gradiente(fx, x)))
        
        # actualizamos las variables de paro
        iteraciones += 1
        cnpo = np.concatenate((gradiente_Lagrangeano_x(fx, hx, x, multiplicadores_Lagrange), hx(x)))
        # if iteraciones % 20 == 0:
        #     print(f"it {iteraciones}, cnpo norm:{np.linalg.norm(cnpo)}")
    return x, multiplicadores_Lagrange, np.linalg.norm(cnpo), fx(x), iteraciones

def funcion_merito(f, h, x_k, c_merito):
    return f(x_k) + c_merito * np.linalg.norm(h(x_k), 1)

# Derivada de la funcion de merito en la direccion p
# respecto a x evaluada en x_k
def derivada_merito_dir_p(f, h, x, c, p):
    return np.dot(gradiente(f, x) , p) - c * np.linalg.norm(h(x), 1)

# obtiene una constance C, de forma que la direccion p 
# sea una direccion de descenso para la funcion de merito
#   f(x) - C ||h(x)||_1 
def constante_funcion_merito(f, h, x_k, p_k, iter, C_max = 100):
    dotGradFP = np.dot(gradiente(f, x_k) , p_k)
    if dotGradFP <= 0:
        C = 10
    else:
        C = min(100, 1 + dotGradFP / np.linalg.norm(h(x_k), 1))
    return C

# Algoritmo auxiliar para la funcion de busqueda en linea
# Tambien sacado del libro del Nocedal
def zoom_busqueda_linea(phi, dphi, c1, c2, a_low, a_high, max_iter = 20):
    for iter in range(max_iter):
        a_j = (a_low + a_high)/2
        phi_aj = phi(a_j)
        if phi_aj > phi(0) + c1*a_j*dphi(0) or phi_aj >= phi(a_low):
            a_high = a_j
        else:
            dphi_aj = dphi(a_j)
            if abs(dphi_aj) <= -c2*dphi(0):
                return a_j
            elif dphi_aj*(a_high - a_low) >= 0:
                a_high = a_low
            a_low = a_j
    return (a_low+a_high)/2

# Algoritmo de busqueda en linea para que cumpla las condiciones de Wolfe
# Tomado del Capitulo 3 del Nocedal
def busqueda_en_linea(f, h, phi, x_k, p_k, C_merito, c1 = 1e-4, c2 = 0.9, max_iters = 10):
    dphi = lambda alpha_x: derivada(phi, alpha_x)
    alpha_max = 1
    alpha = 0.99
    last_alpha = 0
    for iter in range(1, max_iters):
        phi_alpha = phi(alpha) 
        dphi_zero = dphi(0)
        if phi_alpha > phi(0) + c1*alpha*dphi_zero or (iter > 1 and phi_alpha >= phi(last_alpha)):
            return zoom_busqueda_linea(phi, dphi, c1, c2, last_alpha, alpha)
        else:
            dphi_alpha = dphi(alpha)
            if abs(dphi_alpha) <= -c2 * dphi_zero:
                return alpha
            elif dphi_alpha >= 0:
                return zoom_busqueda_linea(phi, dphi, c1, c2, alpha, last_alpha)
            else:
                last_alpha = alpha
                alpha = (alpha+alpha_max) / 2
    return alpha

def busqueda_en_linea_2(phi, c1=1e-4):
    alpha = 1
    dphi = lambda alpha_x: derivada(phi, alpha_x)
    while phi(alpha) > phi(0) + alpha * c1 * dphi(0):
        alpha = alpha / 2
    return alpha

# Gradiente del Lagrangeano respecto a x en x_k
def gradiente_Lagrangeano_x(f, h, x_k, multiplicadores_k):
    return gradiente(f, x_k) + np.dot(jacobiana(h, x_k).T, multiplicadores_k)

def L_BFGS(B, f, h, x_k, C_merito, alpha_k, p_k, multiplicadores_Lagrange):
    s = alpha_k*p_k
    # y = gradiente_Lagrangeano_x(f, h, x_k + s, multiplicadores_Lagrange) - gradiente_Lagrangeano_x(f, h, x_k, multiplicadores_Lagrange)
    fm = lambda x_ : funcion_merito(f, h, x_, C_merito)
    y = gradiente(fm, x_k+s) - gradiente(fm, x_k)
    sBs = np.dot(np.dot(s.T, B), s)
    if np.dot(s.T , y) <= 0.2 * sBs  :
        theta = 0.8 + sBs / (sBs - np.dot(s.T , y))
        r = theta * y + (1 - theta) * np.dot(B , s)
    else:
        r = y
    
    BssB = np.dot(np.dot(B , s) , np.dot(s.T , B))
    B = B + (np.dot(r.T , r) / np.dot(s.T , r)) - (BssB) / (sBs)
    if np.linalg.cond(B) > 10e4:
        B = np.eye(len(x_k))
    return B

def BFGS(B_k, f, h, x_k, C_merito, alpha_k, p_k):
    s_k = alpha_k*p_k
    fm = lambda x_ : funcion_merito(f, h, x_, C_merito)
    y_k = gradiente(fm, x_k+s_k) - gradiente(fm, x_k)
    ys = np.dot(y_k.T, s_k)
    A = np.outer(y_k, y_k) / ys
    Bs = np.dot(B_k, s_k)
    B = B_k + A - np.outer(Bs, Bs.T) / np.dot(s_k.T, Bs)
    if np.linalg.cond(B) > 10e4:
        B = np.eye(len(x_k))
    return B
