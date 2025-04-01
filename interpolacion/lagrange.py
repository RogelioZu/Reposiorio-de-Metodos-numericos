import numpy as np
def interpolacion_lagrange(x, puntos):
    """
    Realiza la interpolación de Lagrange para un valor de x dado.
    
    Parámetros:
    x -- valor para el cual interpolar
    puntos -- lista de tuplas (x_i, y_i) con los puntos conocidos
    
    Retorna:
    El valor interpolado en x
    """
    n = 2# Grado del polinomio
    resultado = 0.0
    
    for i in range(n + 1):
        xi, yi = puntos[i]
        Li = 1.0
        
        for j in range(n + 1):
            if i != j:
                xj, _ = puntos[j]
                Li *= (x - xj) / (xi - xj)
        
        resultado += yi * Li
    
    return resultado

if __name__ == '__main__':
    # Valores dados
    xi = np.array([2, 2.2, 2.4, 2.6, 2.8, 3])
    fi = np.array([0.5103757, 0.5207843, 0.5104147, 0.4813306, 0.4359160, 0.4067381])

    # Crear lista de tuplas (xi, fi)
    puntos = [(xi[i], fi[i]) for i in range(len(xi))]
    print(interpolacion_lagrange(2.33, puntos))