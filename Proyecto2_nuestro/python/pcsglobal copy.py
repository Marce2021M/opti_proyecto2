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
    
    while iteraciones < maximo_iteraciones and np.linalg.norm(cnpo) >= tol:
        # Resolver el subproblema cuadrático
        pk, L = pc(B, jacobiana(hx, x), gradiente(fx, x), -hx(x)) 
        
        if Dk(fx, hx, x, Ck, pk) < 0:
            Ck = Ck
        else:
            Ck = min(Cmax, abs(np.dot(gradiente(fx, x).T , pk)) / np.linalg.norm(hx(x), 1) + 1)
        
        alpha = 1
        while phi(fx, hx, x + alpha * pk, Ck) > phi(fx, hx, x, Ck) + alpha * c1 * Dk(fx, hx, x, Ck, pk):
            alpha = alpha / 2
        
        aux = x
        x = x + alpha * pk
        s = x - aux
        y = LAk(fx, hx, x, L) - LAk(fx, hx, aux, L)
        sBs = np.dot(np.dot(s.T, B), s)
        if np.dot(s.T , y) <= 0.2 * sBs  :
            theta = 0.8 + sBs / (sBs - np.dot(s.T , y))
            r = theta * y + (1 - theta) * np.dot(B , s)
        else:
            r = y
        
        BssB = np.dot(np.dot(B , s) , np.dot(s.T , B))
        B = B + (np.dot(r.T , r) / np.dot(s.T , r)) - (BssB) / (sBs)
        if np.linalg.cond(B) > 10e4:
            B = np.eye(n)

        jac = jacobiana(hx, x)
        L = np.dot(-np.linalg.inv(np.dot(jac , jac.T)), np.dot(jac, gradiente(fx, x)))
        
        iteraciones += 1
        cnpo = np.concatenate((LAk(fx, hx, x, L), hx(x)))
        print(iteraciones, np.linalg.norm(cnpo), fx(x))
    return x, L, iteraciones

def phi(fx, hx, xk, ck):
    return fx(xk) + ck * np.linalg.norm(hx(xk), 1)

def Dk(fx, hx, xk, ck, pk):
    return np.dot(gradiente(fx, xk).T , pk) - ck * np.linalg.norm(hx(xk), 1)

def LAk(fx, hx, xk, Lk):
    return gradiente(fx, xk) + np.dot(jacobiana(hx, xk).T, Lk)

