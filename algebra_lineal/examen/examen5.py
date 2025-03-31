import numpy as np
from tabulate import tabulate

def cramer(A=None, b=None):
    
    # Si no se proporcionan argumentos, usar los del ejemplo original
    if A is None and b is None:
        # Matriz de coeficientes
        A = np.array([[2, 1, -1],
                      [1, 2, 1],
                      [3, -1, 2]])
        # Vector de términos independientes
        b = np.array([6, 9, 5])
    
    # Convertir entradas a arrays de numpy si no lo son ya
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Validar entradas
    if len(A.shape) != 2:
        return "La matriz de coeficientes debe ser bidimensional"
    
    n_filas, n_cols = A.shape
    if n_filas != n_cols:
        return "La matriz de coeficientes debe ser cuadrada"
    
    if len(b.shape) != 1:
        return "El vector de términos independientes debe ser unidimensional"
    
    if n_filas != len(b):
        return "El vector de términos independientes debe tener la misma longitud que la matriz de coeficientes"
    
    # Calcular el determinante de la matriz de coeficientes
    det = float(np.linalg.det(A))
    
    # Verificar si el determinante es cero (o cercano a cero)
    if np.isclose(det, 0, atol=1e-15):
        return "El sistema no tiene solución única (determinante es aproximadamente cero)"
    
    # Inicializar diccionario de resultados y vector solución
    result = {}
    solution = np.zeros(n_filas)
    
    # Aplicar la regla de Cramer para cada variable
    for i in range(n_filas):
        # Crear una nueva matriz reemplazando la columna i con el vector de términos independientes
        A_i = A.copy()
        A_i[:, i] = b
        
        # Calcular el determinante
        det_i = np.linalg.det(A_i)
        
        # Calcular la variable i
        solution[i] = float(det_i / det)
        result[f"x{i+1}"] = solution[i]
    
    # Agregar el determinante original al resultado
    result["determinante_original"] = det
    
    # Crear tabla formateada con los resultados
    table_data = [[f"x{i+1}", f"{solution[i]:.8f}"] for i in range(n_filas)]
    print("\n" + "="*50)
    print(" "*10 + "SOLUCIÓN POR REGLA DE CRAMER")
    print("="*50)
    print(tabulate(table_data, headers=["Variable", "Valor"], tablefmt="grid", numalign="center"))
    print("-"*50)
    print(f"Determinante del sistema: {det:.8f}")
    print("="*50)
    
    return result

if __name__ == '__main__':
   
    A = np.array([
    [3, 2, -1, 1],
    [2, -3, 4, -1],
    [1, 1, 2, 3],
    [5, -2, 3, -4]
])

    b = np.array([5, -3, 7, 1])
    
    print("\nResolviendo sistema ")
    resultado_9x9 = cramer(A, b)