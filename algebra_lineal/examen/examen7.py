import numpy as np

def eliminacion_gaussiana(A, b, tol=1e-10):
    
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
   
    residuo = np.linalg.norm(np.dot(A_original, x) - b_original)
    return residuo < tol, residuo

if __name__ == '__main__':
    # Ejemplo del sistema original
    A = np.array([
    [1,  2, -3,  4, -1,  1,  5, -2],
    [2, -1,  4, -3,  1, -1,  2,  1],
    [-3, 1,  2, -4,  1,  0, -3,  2],
    [5, -2,  3, -4,  1,  2, -1,  1],
    [-2, 4, -1,  0,  3,  2, -5,  1],
    [1, -1,  2,  3, -1,  1, -2,  4],
    [3,  2, -4,  5,  0, -1,  1, -1],
    [2, -3,  0,  1,  4, -2,  3, -1]
])

    b = np.array([10, -5, 7, 12, 3, -8, 6, -2])

    print("Matriz original A:")
    print(A)
    print("Vector b:")
    print(b)

    # Resolver usando nuestra implementacion mejorada
    x = eliminacion_gaussiana(A, b)
    
    if x is not None:
        print("\nVector solucion x (eliminacion gaussiana mejorada):")
        print(x)
        
        # Verificar soluci�n
        es_precisa, residuo = verificar_solucion(A, x, b)
        print(f"Verificacion: {'Precisa' if es_precisa else 'Imprecisa'}")
        print(f"Residuo: {residuo}")
     
    