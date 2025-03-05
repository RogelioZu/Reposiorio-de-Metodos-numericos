import numpy as np

def eliminacion(A,b):
    n = len(A)

    det = np.linalg.det(A)
    if det < 1e-11:
        return "Determinate 0 "
    
    for i in range(n):
        #metodo de pivoteo
        for j in range(i +1,n):
            factor = A[j][i]/A[i][i]
            for k in range(n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]

    #Sustitucion hacia atras
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] = x[i] / A[i][i]
    return x

if __name__ == '__main__':

    A = np.array([[2,4,6], [4,5,6],[3,1,-2]])
    b = np.array([18,24,4])
    print(eliminacion(A,b))
