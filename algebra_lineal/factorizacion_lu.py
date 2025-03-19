import numpy as np
from scipy.linalg import lu_factor, lu_solve, lu


# Definir el sistema Ax = b
A = np.array([[4, 1, -1], 
            [-1, 3, 2], 
            [1, 1, 5]])
b = np.array([7, 3, 10])

det = np.linalg.det(A)
print("Determinante de la matriz:",det)

# Realizar la factorización LU con lu_factor
lu_piv, piv = lu_factor(A)

# Para mostrar L, U y P, podemos usar la función lu directamente
P, L, U = lu(A)

print("Matriz original A:")
print(A)
print("\nMatriz de permutación P:")
print(P)
print("\nMatriz triangular inferior L:")
print(L)
print("\nMatriz triangular superior U:")
print(U)

# Verificar P*L*U = A
print("\nVerificación P*L*U:")
print(np.dot(P, np.dot(L, U)))

# Resolver el sistema usando lu_solve
x = lu_solve((lu_piv, piv), b)

print("\nSolución del sistema:")
print(x)

# Verificar la solución
print("\nVerificación A·x = b:")
print(np.dot(A, x))
