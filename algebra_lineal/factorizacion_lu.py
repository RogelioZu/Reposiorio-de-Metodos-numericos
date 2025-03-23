import numpy as np
from scipy.linalg import lu_factor, lu_solve, lu


# Definir el sistema Ax = b
A = np.array([
    [2, 3, -1, 2, 0, 0, 1, 0, 0],
    [1, 2, 3, 0, -1, 0, 0, 1, 0],
    [0, 1, 2, 3, 0, -1, 0, 0, 1],
    [0, 0, 1, 2, 3, 0, -1, 0, 0],
    [1, 0, 0, 1, 2, 3, 0, -1, 0],
    [0, 1, 0, 0, 1, 2, 3, 0, -1],
    [-1, 0, 1, 0, 0, 1, 2, 3, 0],
    [0, -1, 0, 1, 0, 0, 1, 2, 3],
    [3, 0, -1, 0, 1, 0, 0, 1, 2]
])

b = np.array([5, 8, 9, 7, 10, 6, 11, 12, 9])

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
