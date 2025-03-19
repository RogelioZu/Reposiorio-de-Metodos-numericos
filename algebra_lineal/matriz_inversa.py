import numpy as np

#Metodo para sacar la inversa de una matriz (cuadrada)
def inversa(A):
    
    inv = np.linalg.inv(A)
    print(inv)
    return inv

#Para resolver un sistema de ecuaciones usando esa inversa
#recibimos como paramtros la matriz y el vector de terminos independientes
def resolver_sistema_inversa(A, b):
    det = np.linalg.det(A)
    print(f"determinante de la matriz {det}")
    #Vemos si el determinante de la matriz es diferente de 0
    if abs(det) < 1e-10:
        return "La matriz no tiene inversa determinante 0"
    #Si su det es diferente de cero entonces tiene solucion unica
    inv = inversa(A)
    x = np.dot(inv,b)

    return  x
    

if __name__ == '__main__':
    
   # Matriz de ejemplo 9x9
    A = np.array([
    [4, 1, 0, 0, 0, 0, 0, 0, 2],
    [1, 5, 2, 0, 0, 0, 0, 0, 0],
    [0, 2, 6, 3, 0, 0, 0, 0, 0],
    [0, 0, 3, 7, 4, 0, 0, 0, 0],
    [0, 0, 0, 4, 8, 5, 0, 0, 0],
    [0, 0, 0, 0, 5, 9, 6, 0, 0],
    [0, 0, 0, 0, 0, 6, 10, 7, 0],
    [0, 0, 0, 0, 0, 0, 7, 11, 8],
    [2, 0, 0, 0, 0, 0, 0, 8, 12]
], dtype=float)

# Vector de términos independientes
    b = np.array([7, 8, 10, 14, 17, 20, 23, 26, 22], dtype=float)


    print("Solucion del sistema: ", resolver_sistema_inversa(A,b))