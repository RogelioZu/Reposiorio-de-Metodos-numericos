import numpy as np

def gauss_jordan(A, b):
    """
    Resuelve un sistema de ecuaciones lineales usando el método de eliminación de Gauss-Jordan.
    
    Parámetros:
    A (list o numpy.ndarray): Matriz de coeficientes
    b (list o numpy.ndarray): Vector de términos independientes
    
    Retorna:
    numpy.ndarray: Vector solución
    """
    # Convertir a arrays de numpy y crear la matriz aumentada
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    aumentada = np.hstack((A, b))
    
    n = len(A)
    
    # Realizar la eliminación de Gauss-Jordan
    for i in range(n):
        # Encontrar el elemento máximo en la columna actual (pivoteo parcial)
        max_fila = i
        for k in range(i + 1, n):
            if abs(aumentada[k, i]) > abs(aumentada[max_fila, i]):
                max_fila = k
        
        # Intercambiar la fila máxima con la fila actual
        if max_fila != i:
            aumentada[[i, max_fila]] = aumentada[[max_fila, i]]
        
        # Escalar la fila del pivote para hacer que el elemento pivote sea 1
        pivote = aumentada[i, i]
        aumentada[i] = aumentada[i] / pivote
        
        # Eliminar todas las demás entradas en la columna actual
        for j in range(n):
            if j != i:
                factor = aumentada[j, i]
                aumentada[j] = aumentada[j] - factor * aumentada[i]
    
    # Extraer la solución
    return aumentada[:, -1]

# Ejemplo de uso
if __name__ == "__main__":
    A = np.array([
    [10.2,  2.8,  0.7],   
    [ 1.5,  5,  1.8],
    [ 2.4,  3.5,  10],
    ])
    
    b = np.array([0.17, -8.7, 6.3], dtype=float)
    
    solucion = gauss_jordan(A, b)
    print(f"Solución: {solucion}")