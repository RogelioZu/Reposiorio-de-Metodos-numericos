import numpy as np

def eliminacion_pivoteo_parcial(A, b):
    n = len(A)
    det = np.linalg.det(A)
    if abs(det) < 1e-11:
        print("Determinante cercano a 0 o negativo", det)
        return None
    
    # Crear una matriz aumentada [A|b]
    Ab = np.column_stack((A, b))
    
    for i in range(n):
        # Implementación del pivoteo parcial
        max_index = i
        max_value = abs(Ab[i, i])
        
        # Buscar el elemento con mayor valor absoluto en la columna i
        for k in range(i + 1, n):
            if abs(Ab[k, i]) > max_value:
                max_value = abs(Ab[k, i])
                max_index = k
        
        # Intercambiar filas si es necesario
        if max_index != i:
            Ab[[i, max_index]] = Ab[[max_index, i]]
        
        # Verificar si el elemento pivote es cercano a cero
        if abs(Ab[i, i]) < 1e-10:
            print(f"Error: Elemento pivote en la fila {i} es cercano a cero.")
            return None
        

        
        #Proceso de eliminacion
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            for k in range(i, n + 1):  # n+1 para incluir la columna b
                Ab[j, k] -= factor * Ab[i, k]
    
    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, n]
        for j in range(i + 1, n):
            x[i] -= Ab[i, j] * x[j]
        x[i] = x[i] / Ab[i, i]
    
    return x

if __name__ == '__main__':
    A = np.array([
    [20, 2, 1, -3, 1, 2, -1, 0, 1],
    [3, 25, 2, 1, -2, 0, 1, -1, 0],
    [1, 3, 22, 2, 1, -2, 0, 1, -1],
    [-2, 1, 3, 28, 2, 1, -1, 0, 1],
    [0, -2, 1, 4, 30, 2, 1, -1, 0],
    [1, 0, -2, 1, 3, 26, 2, 1, -1],
    [-1, 1, 0, -2, 1, 3, 24, 2, 1],
    [0, -1, 1, 0, -2, 1, 3, 22, 2],
    [1, 0, -1, 1, 0, -2, 1, 3, 21]
])
    
    b = np.array([100, 120, 110, 130, 140, 125, 115, 105, 95])
    
    #print("Matriz original A:")
    #print(A)
    #print("Vector b:")
    #print(b)
    
    x = eliminacion_pivoteo_parcial(A.copy(), b.copy())
    
    if x is not None:
        print("\nVector Solución del sistema:", x)
        
        # Verificación de la solución
        print("\nVerificación: A·x ≈ b")
        print("A·x =", np.dot(A, x))
        print("b   =", b)
        print("Error relativo:", np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b))