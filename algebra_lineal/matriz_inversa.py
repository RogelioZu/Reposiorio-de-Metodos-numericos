import numpy as np

def inversa(A):
    
    inv = np.linalg.inv(A)
    return inv

def resolver_sistema_inversa(A, b):
    det = np.linalg.det(A)
    print(f"determinante de la matriz {det}")
    #Vemos si el determinante de la matriz es diferente de 0
    if abs(det) < 1e-10:
        return "La matriz no tiene inversa determinante 0"
    inv = inversa(A)
    x = np.dot(inv,b)

    return  x
    

if __name__ == '__main__':

    A = np.array([[3, 2, 1],
                [2, -1, 3],
                [1, 1, -1]], dtype=float)
    print("Inversa de la matriz",inversa(A))
    b = np.array([3.28, 2, 0.41], dtype=float)

    print("Solucion del sistema: ", resolver_sistema_inversa(A,b))