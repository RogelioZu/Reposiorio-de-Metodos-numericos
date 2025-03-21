import numpy as np

def eigen(A):
    return np.linalg.eig(A)

if __name__ == '__main__':
    A = np.array([
    [10,  2,  0,  1,  0,  0,  0,  0,  3],
    [ 2,  8,  4,  0,  0,  0,  0,  0,  0],
    [ 0,  4,  6,  5,  0,  0,  0,  0,  0],
    [ 1,  0,  5, 12,  3,  0,  0,  0,  0],
    [ 0,  0,  0,  3,  9,  1,  0,  0,  0],
    [ 0,  0,  0,  0,  1,  7,  2,  0,  0],
    [ 0,  0,  0,  0,  0,  2, 15,  4,  0],
    [ 0,  0,  0,  0,  0,  0,  4, 11,  5],
    [ 3,  0,  0,  0,  0,  0,  0,  5, 13]
])
    
    valores, vectores = eigen(A)
    
    print("Valores caracter�sticos:")
    for i, valor in enumerate(valores):
        print(f"\u03bb{i+1} = {valor}")
    
    print("\nVectores caracter�sticos:")
    for i in range(len(valores)):
        print(f"Vector v{i+1} (para \u03bb{i+1} = {valores[i]}):")
        print(vectores[:, i])
        
   