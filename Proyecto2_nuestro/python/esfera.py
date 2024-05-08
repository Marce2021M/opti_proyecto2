import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from fesfera import fesfera
from hesfera import hesfera
from pcsglobal import pcsglobal


np.random.seed(0)

n = 20  # número de puntos a generar
x = np.random.rand(60) # llenado de los demás puntos utilizando números aleatorios
f = fesfera
h = hesfera

tic = time.time()
[x, L, iteraciones] = pcsglobal(f, h, x)  # utilizar el método para calcular los puntos.
toc = time.time()
duration = toc - tic

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

P1 = np.zeros((3, n+1))
P1[0,0] = 1
P1[0,1] = 0
P1[0,2] = 0
for j in range(1, n+1):
    P = x[3*(j-1):3*j]
    P1[:, j] = P
    ax.scatter(P[0], P[1], P[2], c='r', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# Genera la Tabla de los Puntos Generados
print('Puntos Generados:')
print('\t\t\t{:>5s} \t\t\t{:>5s} \t\t\t{:>5s}'.format('x', 'y', 'z'))
print('\t-------------------------------------------------------------')

for j in range(n):
    print('\t{:>5.4f} \t\t{:>5.4f} \t\t{:>5.4f}'.format(P1[0, j], P1[1, j], P1[2, j]))

print('\t-------------------------------------------------------------')
