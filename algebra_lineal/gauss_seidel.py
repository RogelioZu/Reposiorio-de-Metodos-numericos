import numpy as np
from tabulate import tabulate

#parecido al metodo de jacobi
def gauss_seidel(A,b,max_iter,x0 =None , tol= 1e-11):
    resultados = []
    n = len(b)
    
    if x0 is None:
        x0 = np.zeros(n)

    #Vectores para guardar la solucion
    x_nuevo = np.copy(x0)
    x_anterior = np.copy(x0)
    #inicializamos el error y contador en 0
    error = 0
    cont = 0
    for i in range(n):
        diagonal = abs(A[i,i])
        suma_fila = 0
        for j in range(n):
            if i != j:
                suma_fila += abs(A[i][j])
        if diagonal >= suma_fila:
            cont += 1  
    if cont == 3:
            print("Matriz Diagonalmente dominante")
    else:
            print("otro caso")
            


    for i in range(max_iter):

        resultados.append([i, x_nuevo[0], x_nuevo[1], x_nuevo[2], error])
        # Calcular la nueva aproximación: x^(k+1) = D^(-1) * (b - R * x^(k))
        #Bueno de otra forma
        #x_nuevo = np.dot(D_inv, (b - np.dot(R, x_anterior)))
        #Aqui viene el cambio con respecto al de jacobi, envez de usar x anterior usamos x nuevo 
        x_nuevo[0] = (b[0]- A[0][1]* x_nuevo[1] - A[0][2]* x_nuevo[2])/A[0][0]
        x_nuevo[1] = (b[1]- A[1][0]* x_nuevo[0] - A[1][2]* x_nuevo[2])/A[1][1]
        x_nuevo[2] = (b[2]- A[2][0]* x_nuevo[0] - A[2][1]* x_nuevo[1])/A[2][2]


         # Calcular el error (norma de la diferencia)
        error = np.linalg.norm(x_nuevo - x_anterior)
        

         # Verificar convergencia
        if error < tol:
             #Aqui tabulamos bro
            print(tabulate(resultados, headers=["n. iter", "x1", "x2", "x3", "error"],tablefmt="pretty",floatfmt=".6f"))
            return x_nuevo, i + 1, error
        
        # Actualizar x para la próxima iteración
        x_anterior = np.copy(x_nuevo)
    
    # Si se alcanza el número máximo de iteraciones
    #Aqui tabulamos bro
    print(tabulate(resultados, headers=["n. iter", "x1", "x2", "x3"],tablefmt="pretty",floatfmt=".6f"))
    return x_nuevo, max_iter, error

if __name__ == '__main__':

    A = np.array([[17, -2, -3], 
                  [-5, 21, -2], 
                  [-5, -5, 22]])
    
    b = np.array([500, 200, 30])

    solucion, iteraciones, error = gauss_seidel(A, b,30)
    
    print(f"Solución: {solucion}")
    print(f"Iteraciones: {iteraciones}")
    print(f"Error final: {error}")