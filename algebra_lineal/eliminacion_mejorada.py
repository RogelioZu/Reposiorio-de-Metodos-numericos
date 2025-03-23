import numpy as np

def eliminacion_gaussiana(A, b, tol=1e-10):
    """
    Resuelve un sistema de ecuaciones lineales Ax = b usando eliminaci�n gaussiana
    con pivoteo parcial para mejorar la estabilidad num�rica.
    
    Par�metros:
    A (numpy.ndarray): Matriz de coeficientes (n x n)
    b (numpy.ndarray): Vector de t�rminos independientes (n)
    tol (float): Tolerancia para detectar singularidades
    
    Retorna:
    numpy.ndarray: Vector soluci�n x, o None si la matriz es singular
    """
    # Crear copias para no modificar los originales
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = len(A)
    
    # Creamos vector para seguir los intercambios de filas (para debug)
    indices = np.arange(n)
    
    # Eliminaci�n hacia adelante con pivoteo parcial
    for i in range(n):
        # Pivoteo parcial: encuentra el mayor elemento en la columna actual
        pivot_row = i + np.argmax(np.abs(A[i:, i]))
        
        # Si el pivote m�ximo es casi cero, la matriz es singular
        if np.abs(A[pivot_row, i]) < tol:
            print(f"Matriz singular o mal condicionada. Elemento diagonal {i+1} casi cero.")
            return None
        
        # Intercambiar filas si es necesario
        if pivot_row != i:
            A[[i, pivot_row]] = A[[pivot_row, i]]  # Intercambio vectorizado
            b[[i, pivot_row]] = b[[pivot_row, i]]
            indices[[i, pivot_row]] = indices[[pivot_row, i]]
            
        # Eliminar elementos debajo del pivote (vectorizado para cada fila)
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]  # Operaci�n vectorizada
            b[j] -= factor * b[i]
    
    # Sustituci�n hacia atr�s (no completamente vectorizable)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    
    return x

def verificar_solucion(A_original, x, b_original, tol=1e-8):
    """
    Verifica la precisi�n de la soluci�n calculada.
    
    Par�metros:
    A_original (numpy.ndarray): Matriz original de coeficientes
    x (numpy.ndarray): Vector soluci�n calculado
    b_original (numpy.ndarray): Vector original de t�rminos independientes
    tol (float): Tolerancia para la verificaci�n
    
    Retorna:
    bool: True si la soluci�n es precisa dentro de la tolerancia
    """
    residuo = np.linalg.norm(np.dot(A_original, x) - b_original)
    return residuo < tol, residuo

if __name__ == '__main__':
    # Ejemplo del sistema original
    A = np.array([
    [2, 3, -1, 2, 0, 0, 1, 0, 0],
    [1, 2, 3, 0, -1, 0, 0, 1, 0],
    [0, 1, 2, 3, 0, -1, 0, 0, 1],
    [0, 0, 1, 2, 3, 0, -1, 0, 0],
    [1, 0, 0, 1, 2, 3, 0, -1, 0],
    [0, 1, 0, 0, 1, 2, 3, 0, -1],
    [-1, 0, 1, 0, 0, 1, 2, 3, 0],
    [0, -1, 0, 1, 0, 0, 1, 2, 3],
    [3, 0, -1, 0, 1, 0, 0, 1, 2]
])
    
    b = np.array([5, 8, 9, 7, 10, 6, 11, 12, 9])

    print("Matriz original A:")
    print(A)
    print("Vector b:")
    print(b)

    # Resolver usando nuestra implementaci�n mejorada
    x = eliminacion_gaussiana(A, b)
    
    if x is not None:
        print("\nVector soluci�n x (eliminaci�n gaussiana mejorada):")
        print(x)
        
        # Verificar soluci�n
        es_precisa, residuo = verificar_solucion(A, x, b)
        print(f"Verificaci�n: {'Precisa' if es_precisa else 'Imprecisa'}")
        print(f"Residuo: {residuo}")
        
        # Comparar con NumPy para referencia
        x_numpy = np.linalg.solve(A, b)
        print("\nVector soluci�n x (numpy.linalg.solve para comparaci�n):")
        print(x_numpy)
        print(f"Diferencia m�xima: {np.max(np.abs(x - x_numpy))}")
    
    