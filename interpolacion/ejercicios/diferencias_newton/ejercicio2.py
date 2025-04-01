import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math

# Procedimiento
# Tabla de diferencias finitas
def tabla_diferencias_finitas(x, y):
    n = len(y)
    diferencias = np.zeros((n, n))
    diferencias[:, 0] = y
    
    # Calcular la diferencias finitas
    for j in range(1, n):
        for i in range(n - j):
            diferencias[i, j] = diferencias[i + 1, j - 1] - diferencias[i, j - 1]
    
    # Crear encabezados para las columnas
    headers = ['i', 'xi', 'f(xi)']
    for j in range(1, n):
        headers.append(f'Δ^{j}f')
    
    # Construir tabla completa
    tabla = np.zeros((n, n+2))
    tabla[:, 0] = np.arange(n)  # Índice/iteración
    tabla[:, 1] = x  # Valores de x
    tabla[:, 2] = y  # Valores de y (f(x))
    
    # Añadir diferencias a la tabla
    for j in range(1, n):
        for i in range(n - j):
            tabla[i, j+2] = diferencias[i, j]
    
    return tabla, headers

def imprimir_tabla(tabla, headers):
    """
    Imprime la tabla de diferencias finitas con encabezados apropiados.
    """
    n = len(tabla)
    
    # Imprimir encabezados
    print('\t'.join(headers))
    print('-' * len('\t'.join(headers)))
    
    # Imprimir filas
    for i in range(n):
        row = [f"{int(tabla[i,0])}", f"{tabla[i,1]:.4f}", f"{tabla[i,2]:.4f}"]
        for j in range(3, len(headers)):
            if i < n - (j-2):
                row.append(f"{tabla[i, j]:.4f}")
            else:
                row.append("")
        print('\t'.join(row))

def polinomio_diferencias_hacia_adelante(x, y, xo):
    n = len(x)
    
    # Encontrar el índice del valor de x que es menor o igual a xo
    idx = 0
    for i in range(n-1, -1, -1):
        if x[i] <= xo:
            idx = i
            break
    
    # Obtener la tabla de diferencias finitas
    tabla, headers = tabla_diferencias_finitas(x, y)
    
    # Variable simbólica para el polinomio
    X = sp.symbols('x')
    
    # Valor inicial para el polinomio (primer término)
    f_xo = tabla[idx, 2]
    
    # Calcular el paso h
    h = x[1] - x[0]
    
    # Para calcular el término (x-x0)/h, (x-x0)(x-x1)/2!h^2, etc.
    s = (X - x[idx]) / h
    
    # Inicializar polinomio con el primer término
    polinomio = sp.sympify(f_xo)
    
    print(f"\nAproximaciones iterativas en x = {xo}:")
    print(f"Paso 0: P0(x) = {f_xo}")
    print(f"Evaluado en x = {xo}: {f_xo}")
    
    # Construir el polinomio término a término
    factorial = 1
    producto = 1
    ultimo_coef = 0  # Para mantener el último coeficiente como en el original
    
    for j in range(1, n):
        if idx + j - 1 < n - j:  # Verificar límites
            # Obtener la diferencia finita correspondiente
            diff = tabla[idx, j+2]
            ultimo_coef = diff  # Guardar el último coeficiente
            
            # Construir el producto (x-x0)(x-x1)...(x-x_{j-1})
            if j > 1:
                producto *= (s - (j-1))
            else:
                producto = s
            
            # Calcular el factorial para el denominador
            factorial *= j
            
            # Añadir el nuevo término al polinomio
            termino = diff * producto / factorial
            polinomio += termino
            
            # Imprimir la fórmula desarrollada hasta este paso
            polinomio_expandido = sp.expand(polinomio)
            print(f"\nPaso {j}: P{j}(x) = P{j-1}(x) + {diff} * {producto} / {factorial}")
            print(f"P{j}(x) = {polinomio_expandido}")
            print(f"Evaluado en x = {xo}: {float(polinomio_expandido.subs(X, xo))}")
    
    # Expandir y simplificar el polinomio final
    polinomio_final = sp.expand(polinomio)
    
    # Regresar los mismos valores que la función original
    return str(polinomio_final), tabla, headers, ultimo_coef


    


if __name__ == '__main__':
    xi = np.array([2, 2.2, 2.4, 2.6, 2.8, 3])
    fi = np.array([0.5103757, 0.5207843, 0.5104147, 0.4813306, 0.4359160, 0.4067381])
    polinomio, table, head, coeficientes = polinomio_diferencias_hacia_adelante(xi, fi, 2.89)
   
    
    print("\nTabla de diferencias finitas:")
    imprimir_tabla(table, head)
    
    print("\nPolinomio final de interpolación:")
    print(polinomio)
    