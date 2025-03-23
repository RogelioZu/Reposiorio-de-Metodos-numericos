import numpy as np
from tabulate import tabulate

def jacobi(A, b, max_iter, x0=None, tol=1e-11):
   
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
    cont = 0
    for i in range(n):
        diagonal = abs(A[i, i])
        suma_fila = sum(abs(A[i, j]) for j in range(n) if i != j)
        if diagonal >= suma_fila:
            cont += 1
    
    if cont == n:
        print("Matriz diagonalmente dominante")
    else:
        print("Matriz no es diagonalmente dominante (puede que no converja)")
    
    # Para matrices grandes, mostraremos solo algunas variables en la tabulación
    if n > 5:
        # Mostrar primeras 2, últimas 2, y el error
        headers = ["Iter"] + [f"x₁", f"x₂", "...", f"x{n-1}", f"x{n}"] + ["Error"]
    else:
        # Mostrar todas las variables si son pocas
        headers = ["Iter"] + [f"x{i+1}" for i in range(n)] + ["Error"]
    
    # Bucle principal del método de Jacobi
    for iteracion in range(max_iter):
        # Guardar solo las iteraciones importantes para no saturar la tabla
        # Primera iteración, últimas 10 iteraciones, o cada 5 iteraciones si son muchas
        if iteracion == 0 or iteracion >= max_iter - 10 or iteracion % 5 == 0:
            # Para matrices grandes, mostrar solo algunas variables seleccionadas
            if n > 5:
                fila = [iteracion]
                fila.append(x_nuevo[0])          # Primera variable
                fila.append(x_nuevo[1])          # Segunda variable
                fila.append("...")               # Indicador de variables omitidas
                fila.append(x_nuevo[n-2])        # Penúltima variable
                fila.append(x_nuevo[n-1])        # Última variable
                fila.append(error)
            else:
                # Si hay pocas variables, mostrar todas
                fila = [iteracion] + list(x_nuevo) + [error]
            
            resultados.append(fila)
        
        # Implementación del método de Jacobi (diferencia clave con Gauss-Seidel)
        # Todos los cálculos usan valores de la iteración anterior
        for i in range(n):
            suma = 0
            for j in range(n):
                if i != j:
                    suma += A[i, j] * x_anterior[j]
            
            x_nuevo[i] = (b[i] - suma) / A[i, i]
        
        # Calcular el error (norma de la diferencia)
        error = np.linalg.norm(x_nuevo - x_anterior)
        
        # Verificar convergencia
        if error < tol:
            print("\n" + "="*80)
            print(" "*30 + "RESULTADOS DE LAS ITERACIONES")
            print("="*80)
            print(tabulate(resultados, headers=headers, 
                          tablefmt="grid", 
                          floatfmt=".8f", 
                          numalign="center"))
            print("="*80)
            print(f"¡Convergencia alcanzada en {iteracion+1} iteraciones!")
            print("="*80 + "\n")
            return x_nuevo, iteracion + 1, error
        
        # Actualizar x para la próxima iteración
        x_anterior = np.copy(x_nuevo)
    
    # Si se alcanza el número máximo de iteraciones
    print("\n" + "="*80)
    print(" "*30 + "RESULTADOS DE LAS ITERACIONES")
    print("="*80)
    print(tabulate(resultados, headers=headers, 
                  tablefmt="grid", 
                  floatfmt=".8f", 
                  numalign="center"))
    print("="*80)
    print(f"Se alcanzó el máximo de {max_iter} iteraciones sin convergencia.")
    print("="*80 + "\n")
    return x_nuevo, max_iter, error

if __name__ == '__main__':
    # Ejemplo original
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
    
    print("\n" + "*"*50)
    print(" "*10 + "RESOLVIENDO SISTEMA ORIGINAL")
    print("*"*50)
    
    solucion, iteraciones, error = jacobi(A, b, 30)
    
    print("\n" + "-"*70)
    print(" "*20 + "RESULTADOS FINALES")
    print("-"*70)
    
    # Formatear la solución en una tabla bien presentada
    sol_table = []
    for i in range(len(solucion)):
        sol_table.append([f"x{i+1}", f"{solucion[i]:.8f}"])
    
    print(tabulate(sol_table, headers=["Variable", "Valor"], 
                  tablefmt="grid", numalign="center"))
    
    print("\n" + "-"*70)
    print(f"Total de iteraciones: {iteraciones}")
    print(f"Error final: {error:.10e}")