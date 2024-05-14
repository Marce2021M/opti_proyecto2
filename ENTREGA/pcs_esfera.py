"""
Autores: 
    - Diana Espinosa Ruiz CU: 179164
    - Alfredo Alef Pineda Reyes CU: 191164
    - Marcelino Sanchez Rodriguez CU: 191654
    - Carlos Alberto Delgado Elizondo CU: 181866

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from f_electron import f_electron
from h_esfera import h_esfera
from pcs_global import pcs_global

np.random.seed(0)

n = 20  # número de puntos a generar
theta = np.random.uniform(0, 2*np.pi, n)
phi = np.arccos(1 - 2*np.random.uniform(0, 1, n))
P = np.zeros((3, n))
P[0, :] = np.random.rand(1, n)
P[1, :] = np.random.rand(1, n)
for i in range(n):
    while P[0, i]**2 + P[1, i]**2 > 1:
        P[0, i] = P[0, i]/2
    ini = 0
    fin = 1
    for _ in range(100):
        mitad = (ini + fin) /2
        if mitad**2 + P[1,i]**2 + P[0, i]**2  > 1:
            fin = mitad
        else:
            ini = mitad
    P[2, i] = ini

x = P.T.flatten()
f = f_electron
h = h_esfera

tic = time.time()
[x, _, cnpo_norm, fx, iteraciones] = pcs_global(f, h, x)  # utilizar el método para calcular los puntos.
toc = time.time()
duration = toc - tic

print(f"np:{n+1} \tCNPO:{cnpo_norm} \tf(x*):{fx} \tcpu time:{duration}s")

P = np.zeros((3, n+1))
P[0,0] = 1
P[0,1] = 0
P[0,2] = 0
for j in range(1, n+1):
    P[:, j] = x[3*(j-1):3*j]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1]) 

ax.scatter(P[0, :], P[1, :], P[2, :], color='r', s=50)

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_s = np.outer(np.cos(u), np.sin(v))
y_s = np.outer(np.sin(u), np.sin(v))
z_s = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_s, y_s, z_s, color='b', alpha=0.1)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# Genera la Tabla de los Puntos Generados
print('Puntos Generados:')
print('\t\t\t{:>5s} \t\t\t{:>5s} \t\t\t{:>5s}'.format('x', 'y', 'z'))
print('\t-------------------------------------------------------------')

for j in range(n):
    print('\t{:>5.4f} \t\t{:>5.4f} \t\t{:>5.4f}'.format(P[0, j], P[1, j], P[2, j]))

print('\t-------------------------------------------------------------')
