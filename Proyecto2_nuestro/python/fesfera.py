import numpy as np

def fesfera(x):
    n = len(x)
    num_puntos = n//3
    # Reorganiza x para que cada fila sea un punto en el espacio tridimensional.
    puntos = x.reshape(-1, 3)
    
    # Punto fijo
    punto_fijo = np.array([1, 0, 0])
    
    # Calcula las distancias cuadradas desde el punto fijo
    distancias_a_fijo = np.linalg.norm(puntos - punto_fijo, axis=1)

    # Suma de los inversos de las distancias al punto fijo
    f = np.sum(1 / distancias_a_fijo)

    # Ahora, calcula la suma de las inversas de las distancias entre cada par de puntos
    for i in range(num_puntos):
        for j in range(i + 1, num_puntos):
            # Calcula la distancia cuadrada entre el punto i y el punto j
            distance_between_puntos = np.sum((puntos[i] - puntos[j]) ** 2)
            # Suma la inversa de la raíz cuadrada de la distancia al total
            f += 1 / np.sqrt(distance_between_puntos)

    return f

