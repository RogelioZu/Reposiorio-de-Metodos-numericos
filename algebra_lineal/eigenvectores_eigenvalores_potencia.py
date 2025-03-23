import numpy as np

def metodo_potencia_simple(A, tol=1e-10, max_iter=1000, verbose=False):
   
    # Obtener dimensión de la matriz
    n = A.shape[0]
    
    # Inicializar vector aleatorio y normalizarlo
    x_k = np.random.rand(n)
    x_k = x_k / np.linalg.norm(x_k)
    
    # Valor inicial
    lambda_k = 0
    
    # Historial de valores característicos para analizar convergencia
    historia = []
    
    # Proceso iterativo
    for k in range(max_iter):
        # Guardar valor anterior para comparar
        lambda_anterior = lambda_k
        
        # Multiplicar la matriz por el vector
        y_k = np.dot(A, x_k)
        
        # Calcular aproximación del valor característico (cociente de Rayleigh)
        lambda_k = np.dot(x_k.T, np.dot(A, x_k))
        
        # Normalizar el vector resultante
        x_k = y_k / np.linalg.norm(y_k)
        
        # Guardar valor estimado
        historia.append(lambda_k)
        
        # Mostrar progreso si verbose es True
        if verbose and k % 10 == 0:
            print(f"Iteración {k}: λ = {lambda_k}")
        
        # Verificar convergencia
        if abs(lambda_k - lambda_anterior) < tol:
            if verbose:
                print(f"Convergencia alcanzada en {k+1} iteraciones")
            return lambda_k, x_k, k+1, historia
    
    # Si no converge en max_iter iteraciones
    print(f"Advertencia: No se alcanzó la convergencia después de {max_iter} iteraciones")
    return lambda_k, x_k, max_iter, historia

# Ejemplo de uso
if __name__ == "__main__":
    # Crear una matriz de ejemplo
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
    
    # Establecer una semilla para reproducibilidad
    np.random.seed(42)
    
    # Aplicar el método de la potencia
    valor, vector, iteraciones, historia = metodo_potencia_simple(A, verbose=True)
    
    print("\nResultados:")
    print(f"Valor característico dominante: {valor}")
    print(f"Vector característico asociado: {vector}")
    print(f"Número de iteraciones: {iteraciones}")
    
    # Verificación: comparar con numpy.linalg.eig
    valores_np, vectores_np = np.linalg.eig(A)
    idx_max = np.argmax(np.abs(valores_np))
    
    print("\nResultados con numpy.linalg.eig:")
    print(f"Valor característico dominante: {valores_np[idx_max]}")
    print(f"Vector característico asociado: {vectores_np[:, idx_max]}")
    
    # Verificar Av = λv
    print("\nVerificación Av = λv:")
    Av = np.dot(A, vector)
    lambdav = valor * vector
    print(f"A·v = {Av}")
    print(f"λ·v = {lambdav}")
    print(f"Error: {np.linalg.norm(Av - lambdav)}")
   