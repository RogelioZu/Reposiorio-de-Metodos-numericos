import numpy as np

def cramer():
    # Matriz de coeficientes
    mat = np.array([[2, 1, -1],
               [1, 2, 1],
               [3, -1, 2]])
    # Vector de términos independientes
    sol = np.array([6, 9, 5])
    
    # Calcular el determinante de la matriz de coeficientes
    det = float(np.linalg.det(mat))
    
    # Verificar si el determinante es cero (en cuyo caso no se puede aplicar la regla de Cramer)
    if np.isclose(det, 0):
        return "El sistema no tiene solución única (determinante es cero)"
    
    # Crear matrices para cada variable reemplazando la columna correspondiente con el vector solución
    # Matriz para x (reemplazar primera columna)
    mat_x = mat.copy()
    mat_x[:, 0] = sol
    
    # Matriz para y (reemplazar segunda columna)
    mat_y = mat.copy()
    mat_y[:, 1] = sol
    
    # Matriz para z (reemplazar tercera columna)
    mat_z = mat.copy()
    mat_z[:, 2] = sol
    
    # Calcular determinantes
    det_x = np.linalg.det(mat_x)
    det_y = np.linalg.det(mat_y)
    det_z = np.linalg.det(mat_z)
    
    # Aplicar la regla de Cramer para encontrar los valores de x, y, z
    x = float(det_x / det)
    y = float(det_y / det)
    z = float(det_z / det)
    
    return {"x": x, "y": y, "z": z, "determinante_original": det}

if __name__ == '__main__':
    resultado = cramer()
    print(resultado)