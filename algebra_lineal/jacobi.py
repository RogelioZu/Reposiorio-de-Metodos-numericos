
import numpy as np

def jacobi(A,b,max_iter,x0 =None , tol= 1e-11):

    n = len(b)
    
    if x0 is None:
        x0 = np.zeros(n)

    #Vectores para guardar la solucion
    x_nuevo = np.copy(x0)
    x_anterior = np.copy(x0)

    #Extraer la diagonal de A
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    R = A - D

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

        # Calcular la nueva aproximación: x^(k+1) = D^(-1) * (b - R * x^(k))
        x_nuevo = np.dot(D_inv, (b - np.dot(R, x_anterior)))

         # Calcular el error (norma de la diferencia)
        error = np.linalg.norm(x_nuevo - x_anterior)
        

         # Verificar convergencia
        if error < tol:
            return x_nuevo, i + 1, error
        
        # Actualizar x para la próxima iteración
        x_anterior = np.copy(x_nuevo)
    
    # Si se alcanza el número máximo de iteraciones
    return x_nuevo, max_iter, error

if __name__ == '__main__':

    A = np.array([[4, 1, -1], 
                  [-1, 3, 2], 
                  [1, 1, 5]])
    
    b = np.array([7, 3, 10])

    solucion, iteraciones, error = jacobi(A, b,100)
    
    print(f"Solución: {solucion}")
    print(f"Iteraciones: {iteraciones}")
    print(f"Error final: {error}")