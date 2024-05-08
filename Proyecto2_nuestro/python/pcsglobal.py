import numpy as np
from gradiente import gradiente
from jacobiana import jacobiana
from pc import pc

def pcsglobal(fx, hx, x0):
    n = len(x0)
    m = len(hx(x0))
    tol = 1e-5 # tolerancia
    maximo_iteraciones = 100
    
    c1 = 1e-2 
    C0 = 1
    x = x0
    B = np.eye(n)
    L = np.zeros(m)
    iteraciones = 0
    cnpo = np.concatenate((LAk(fx, hx, x, L), hx(x)))
    Cmax = 10e5
    Ck = C0
    
    while iteraciones <= maximo_iteraciones and np.linalg.norm(cnpo) >= tol:
        # Resolver el subproblema cuadrático
        pk, L = pc(B, jacobiana(hx, x), gradiente(fx, x), -hx(x)) 
        
        if Dk(fx, hx, x, Ck, pk) < 0:
            Ck = Ck
        else:
            Ck = min(Cmax, abs(gradiente(fx, x).T * pk) / np.linalg.norm(hx(x), 1) + 1)
        
        alpha = 1
        while phi(fx, hx, x + alpha * pk, Ck) > phi(fx, hx, x, Ck) + alpha * c1 * Dk(fx, hx, x, Ck, pk):
            alpha = alpha / 2
        
        aux = x
        x = x + alpha * pk
        s = x - aux
        y = LAk(fx, hx, x, L) - LAk(fx, hx, aux, L)
        if s.T * y <= 0.2 * (s.T * B * s):
            theta = 0.8 * (s.T * B * s) / (s.T * B * s - s.T * y)
            r = theta * y + (1 - theta) * B * s
        else:
            r = y
        
        B = B - (B * s* s.T * B) / (s.T * B * s) + (r.T *r) / (s.T * r)
        if np.linalg.cond(B) > 10e4:
            B = np.eye(n)
        
        L = -np.linalg.inv(jacobiana(hx, x) * jacobiana(hx, x).T) * jacobiana(hx, x) * gradiente(fx, x)
        iter += 1
        cnpo = np.concatenate((LAk(fx, hx, x, L), hx(x)))
    
    return x, L, iter

def phi(fx, hx, xk, ck):
    return fx(xk) + ck * np.linalg.norm(hx(xk), 1)

def Dk(fx, hx, xk, ck, pk):
    return gradiente(fx, xk) * pk - ck * np.linalg.norm(hx(xk), 1)

def LAk(fx, hx, xk, Lk):
    return gradiente(fx, xk) + jacobiana(hx, xk).T * Lk

