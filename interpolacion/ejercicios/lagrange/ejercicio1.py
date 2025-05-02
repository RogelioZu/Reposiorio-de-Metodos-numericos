import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def interpolacion_lagrange(x, y, xi):
    yi = 0
    n = x.size
    
    # Calcula los factores de lagrange y hace la suma
    for i in range(0, n):
        producto = y[i]
        for j in range(0, n):
            if i != j:
                producto = producto * (xi - x[j]) / (x[i] - x[j])
        yi = yi + producto
    
    # Resultado de la interpolación manual
    print(f"Resultado de la interpolación en x = {xi}:")
    print(f"Valor calculado manualmente: {yi}")
    
    # Verificación con la función de SciPy
    f = interpolate.lagrange(x, y)
    valor_scipy = f(xi)
    print(f"Valor usando scipy.interpolate.lagrange: {valor_scipy}")

    # Gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Datos')
    
    # Puntos para la curva
    x_plot = np.linspace(min(x), max(x), 100)
    y_plot = f(x_plot)
    plt.plot(x_plot, y_plot, '-', label='Polinomio de Lagrange')
    
    # Punto interpolado
    plt.plot(xi, yi, 'sr', label='Punto interpolado')
    
    plt.title(f"Interpolación de Lagrange (x={xi}, y={yi:.6f})")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return yi

if __name__ == '__main__':
    # Valores dados
    x = np.array([2, 3, 6, 8, 10, 12])
    y = np.array([0.1, 0.15, 0.2, 0.25, 0.35, 0.4])
    
    # Punto a interpolar
    xi = 5
    
    # Realizar la interpolación y mostrar resultados
    interpolacion_lagrange(x, y, xi)