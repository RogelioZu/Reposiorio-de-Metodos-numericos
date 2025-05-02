import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
import pandas as pd

def hermite_interpolation(x, y, dy, x_interp):
    """
    Realiza interpolación de Hermite.
    
    Parámetros:
    x: puntos x conocidos
    y: valores de función en puntos x
    dy: derivadas en puntos x
    x_interp: puntos x para interpolar
    
    Retorna:
    cs: el objeto spline para evaluar la función y sus derivadas
    """
    # Usar CubicHermiteSpline para interpolar
    cs = CubicHermiteSpline(x, y, dy)
    
    return cs

# Datos proporcionados
x = np.array([1, 2, 4, 7, 9])
y = np.array([3, 6, 8, 10, 5])

# Para la interpolación de Hermite, necesitamos las derivadas
# Como no se proporcionaron, las estimaremos con diferencias finitas
dy = np.zeros_like(x, dtype=float)

# Diferencias finitas para puntos interiores
for i in range(1, len(x)-1):
    dy[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])

# Para los extremos, usamos diferencias hacia adelante y hacia atrás
dy[0] = (y[1] - y[0]) / (x[1] - x[0])  # Derivada en el primer punto
dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])  # Derivada en el último punto

# Crear el objeto spline para poder evaluar la función y sus derivadas
spline = hermite_interpolation(x, y, dy, None)

# Punto específico para interpolar
x_specific = 3
y_specific = spline(x_specific)
print(f"\nValor interpolado en x = {x_specific}: y = {y_specific:.6f}")

# Generar puntos para la interpolación y la gráfica
x_interp = np.linspace(min(x), max(x), 100)

# Realizar la interpolación para la gráfica
y_interp = spline(x_interp)

# Crear tabla con las derivadas
# Vamos a seleccionar algunos puntos para la tabla
selected_x = np.linspace(min(x), max(x), 10)  # 10 puntos equidistantes

# Evaluar la función y sus derivadas en estos puntos
f_x = spline(selected_x)      # Función evaluada
f_x_prime = spline(selected_x, 1)  # Primera derivada
f_x_double_prime = spline(selected_x, 2)  # Segunda derivada

# Crear un DataFrame para mostrar la tabla
table_data = {
    'x': selected_x,
    'f(x)': f_x,
    'f\'(x)': f_x_prime,
    'f\'\'(x)': f_x_double_prime
}
df = pd.DataFrame(table_data)
print("\nTabla de resultados de la interpolación de Hermite:")
print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(x_interp, y_interp, 'b-', label='Interpolación de Hermite')
plt.plot(x, y, 'ro', label='Puntos originales')
plt.plot(x_specific, y_specific, 'g*', markersize=10, label=f'Punto x={x_specific}, y={y_specific:.4f}')
plt.title('Interpolación de Hermite')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Información adicional sobre la interpolación
print("\nInformación de la interpolación:")
print(f"Puntos originales (x): {x}")
print(f"Valores originales (y): {y}")
print(f"Derivadas estimadas (dy): {dy}")