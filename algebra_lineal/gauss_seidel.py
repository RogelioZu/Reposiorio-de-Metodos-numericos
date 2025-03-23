import numpy as np
from tabulate import tabulate

def gauss_seidel(A, b, max_iter, x0=None, tol=1e-11):
   
   
    # Obtener la dimensión del sistema
    n = len(b)
    
    # Inicializar vector de solución inicial
    if x0 is None:
        x0 = np.zeros(n)
    
    # Vectores para guardar la solución
    x_nuevo = np.copy(x0)
    x_anterior = np.copy(x0)
    
    # Inicializamos el error y estructuras para resultados
    error = 0
    resultados = []
    
    # Verificar si la matriz es diagonalmente dominante
    es_dominante = True
    for i in range(n):
        diagonal = abs(A[i, i])
        suma_fila = sum(abs(A[i, j]) for j in range(n) if i != j)
        if diagonal < suma_fila:
            es_dominante = False
            break
    
    if es_dominante:
        print("Matriz diagonalmente dominante")
    else:
        print("Matriz no es diagonalmente dominante (puede que no converja)")
    
    # Crear encabezados dinámicos para la tabla de resultados
    headers = ["n. iter"] + [f"x{i+1}" for i in range(n)] + ["error"]
    
    # Bucle principal del método de Gauss-Seidel
    for iteracion in range(max_iter):
        # Guardar resultado de la iteración actual
        resultado_iter = [iteracion] + list(x_nuevo) + [error]
        resultados.append(resultado_iter)
        
        # Implementación del método de Gauss-Seidel
        for i in range(n):
            suma = 0
            for j in range(n):
                if i != j:
                    suma += A[i, j] * x_nuevo[j]
            
            x_nuevo[i] = (b[i] - suma) / A[i, i]
        
        # Calcular el error (norma de la diferencia)
        error = np.linalg.norm(x_nuevo - x_anterior)
        
        # Verificar convergencia
        if error < tol:
            print(tabulate(resultados, headers=headers, tablefmt="pretty", floatfmt=".6f"))
            return x_nuevo, iteracion + 1, error
        
        # Actualizar x para la próxima iteración
        x_anterior = np.copy(x_nuevo)
    
    # Si se alcanza el número máximo de iteraciones
    print(tabulate(resultados, headers=headers, tablefmt="pretty", floatfmt=".6f"))
    return x_nuevo, max_iter, error

if __name__ == '__main__':
    # Sistema de ecuaciones nxn
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
    
    
    solucion, iteraciones, error = gauss_seidel(A, b, 100)
    
    print(f"\nSolución: {solucion}")
    print(f"Iteraciones: {iteraciones}")
    print(f"Error final: {error}")
    
   