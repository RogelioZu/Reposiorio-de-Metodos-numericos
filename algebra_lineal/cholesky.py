import numpy as np
from tabulate import tabulate

def cholesky_decomposition(A):
    
    A = np.array(A, dtype=float)  # Asegurar que A es un array numpy con tipo float
    n = len(A)
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1):
            if i == j:  # Elementos diagonales
                sum_k = sum(L[i, k] ** 2 for k in range(j))
                value = A[i, i] - sum_k
                if value <= 0:
                    raise ValueError("La matriz no es definida positiva")
                L[i, j] = np.sqrt(value)
            else:  # Elementos no diagonales
                sum_k = sum(L[i, k] * L[j, k] for k in range(j))
                if L[j, j] == 0:
                    raise ValueError("División por cero en la descomposición de Cholesky")
                L[i, j] = (A[i, j] - sum_k) / L[j, j]
                
    return L

def visualize_matrices(A, L, float_format=".6f"):
    
    print("Matriz Original A:")
    print(tabulate(A, tablefmt="grid", floatfmt=float_format))
    
    print("\nDescomposición de Cholesky L:")
    print(tabulate(L, tablefmt="grid", floatfmt=float_format))
    
    # Verificar la descomposición: A = L*L^T
    LLT = np.dot(L, L.T)
    print("\nVerificación L*L^T:")
    print(tabulate(LLT, tablefmt="grid", floatfmt=float_format))
    
    # Comprobar qué tan cercano es LLT a A
    error = np.max(np.abs(A - LLT))
    print(f"\nError Absoluto Máximo: {error:.2e}")

def is_positive_definite(A):
   
    try:
        # Intenta calcular la descomposición de Cholesky usando numpy
        # Esto lanzará un error si A no es definida positiva
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def solve_with_cholesky(L, b, float_format=".6f"):
    """
    Resuelve un sistema de ecuaciones Ax = b usando la factorización de Cholesky ya calculada.
    
    Parámetros:
    L (numpy.ndarray): Matriz triangular inferior de la factorización de Cholesky
    b (numpy.ndarray): Vector del lado derecho
    float_format (str): Formato para números de punto flotante
    
    Retorna:
    numpy.ndarray: Vector solución x
    """
    n = len(b)
    
    print("\n" + "="*70)
    print("RESOLUCIÓN DEL SISTEMA Ax = b USANDO CHOLESKY PRECOMPUTADO")
    print("="*70)
    
    # PASO 1: Mostrar los datos de entrada
    print("\nMatriz triangular inferior L (Cholesky precomputado):")
    print(tabulate(L, tablefmt="grid", floatfmt=float_format))
    
    print("\nVector b:")
    b_table = [[f"b{i+1}", f"{b[i]:{float_format}}"] for i in range(n)]
    print(tabulate(b_table, headers=["Índice", "Valor"], tablefmt="grid"))
    
    # PASO 2: Resolver Ly = b (sustitución hacia adelante)
    print("\n" + "-"*70)
    print("PASO 1: Resolver Ly = b (sustitución hacia adelante)")
    print("-"*70)
    
    # Mostrar el sistema de ecuaciones Ly = b
    print("\nSistema Ly = b:")
    equations_Ly = []
    for i in range(n):
        eq = ""
        for j in range(i+1):  # Solo hasta i porque L es triangular inferior
            if j > 0:
                eq += " + " if L[i, j] >= 0 else " - "
                eq += f"{abs(L[i, j]):{float_format}}·y{j+1}"
            else:
                eq += f"{L[i, j]:{float_format}}·y{j+1}"
        eq += f" = {b[i]:{float_format}}"
        equations_Ly.append([eq])
    print(tabulate(equations_Ly, tablefmt="plain"))
    
    # Resolver Ly = b
    y = np.zeros(n)
    y_steps = []
    
    for i in range(n):
        # Calcular la suma de los términos L[i,j] * y[j] para j < i
        sum_j = sum(L[i, j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_j) / L[i, i]
        
        # Registrar cada paso del cálculo
        step = f"y{i+1} = ({b[i]:{float_format}}"
        if sum_j != 0:
            step += f" - {sum_j:{float_format}}"
        step += f") / {L[i, i]:{float_format}} = {y[i]:{float_format}}"
        y_steps.append([step])
    
    print("\nResolución de y paso a paso:")
    print(tabulate(y_steps, tablefmt="plain"))
    
    print("\nSolución y:")
    y_solution = [["y" + str(i+1), f"{y[i]:{float_format}}"] for i in range(n)]
    print(tabulate(y_solution, headers=["Variable", "Valor"], tablefmt="grid"))
    
    # PASO 3: Resolver L^Tx = y (sustitución hacia atrás)
    print("\n" + "-"*70)
    print("PASO 2: Resolver L^Tx = y (sustitución hacia atrás)")
    print("-"*70)
    
    # Mostrar la matriz L^T
    LT = L.T
    print("\nMatriz L^T:")
    print(tabulate(LT, tablefmt="grid", floatfmt=float_format))
    
    # Mostrar el sistema de ecuaciones L^Tx = y
    print("\nSistema L^Tx = y:")
    equations_LTx = []
    for i in range(n):
        eq = ""
        for j in range(i, n):  # Desde i hasta n-1 porque LT es triangular superior
            if j > i:
                eq += " + " if LT[i, j] >= 0 else " - "
                eq += f"{abs(LT[i, j]):{float_format}}·x{j+1}"
            else:
                eq += f"{LT[i, j]:{float_format}}·x{j+1}"
        eq += f" = {y[i]:{float_format}}"
        equations_LTx.append([eq])
    print(tabulate(equations_LTx, tablefmt="plain"))
    
    # Resolver L^Tx = y
    x = np.zeros(n)
    x_steps = []
    
    for i in range(n-1, -1, -1):  # Desde n-1 hasta 0
        # Calcular la suma de los términos LT[i,j] * x[j] para j > i
        sum_j = sum(LT[i, j] * x[j] for j in range(i+1, n))
        x[i] = (y[i] - sum_j) / LT[i, i]
        
        # Registrar cada paso del cálculo
        step = f"x{i+1} = ({y[i]:{float_format}}"
        if sum_j != 0:
            step += f" - {sum_j:{float_format}}"
        step += f") / {LT[i, i]:{float_format}} = {x[i]:{float_format}}"
        x_steps.append([step])
    
    # Revertir los pasos para mostrarlos en orden desde x1 a xn
    x_steps.reverse()
    
    print("\nResolución de x paso a paso:")
    print(tabulate(x_steps, tablefmt="plain"))
    
    print("\nSolución x:")
    x_solution = [["x" + str(i+1), f"{x[i]:{float_format}}"] for i in range(n)]
    print(tabulate(x_solution, headers=["Variable", "Valor"], tablefmt="grid"))
    
    return x

if __name__ == "__main__":
     
    A = np.array([
    [10.2,  2.8,  0.7],   
    [ 1.5,  5,  1.8],
    [ 2.4,  3.5,  10],
    ])
    
    b = np.array([0.17, -8.7, 6.3], dtype=float)
    

    if is_positive_definite(A):
        L = cholesky_decomposition(A)
        visualize_matrices(A,L)
        x = solve_with_cholesky(L,b)
        print(x)
         # Verificar la solución
        Ax = A @ x
        print("\n" + "="*70)
        print("VERIFICACIÓN FINAL: Ax = b")
        print("="*70)
        
        verification = np.column_stack((Ax, b))
        print(tabulate(verification, headers=["Ax", "b"], tablefmt="grid", floatfmt=".6f"))
        
        # Calcular y mostrar el error
        error = np.linalg.norm(Ax - b)
        print(f"Error (norma de Ax - b): {error:.2e}")
        
        if error < 1e-10:
            print("¡La solución es correcta! (Error muy cercano a cero)")
        else:
            print("Advertencia: La precisión de la solución podría ser mejorable")
    else:
        print("La matriz no es definida positiva, no se puede aplicar la factorización de Cholesky.")