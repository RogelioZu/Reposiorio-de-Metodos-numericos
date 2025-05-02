import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def polynomial_regression_manual(x, y, degree):
    """
    Realiza regresión polinomial manual usando mínimos cuadrados.
    
    Parámetros:
    x: valores independientes
    y: valores dependientes
    degree: grado del polinomio
    
    Retorna:
    coeffs: coeficientes del polinomio (de menor a mayor grado)
    y_pred: valores predichos
    r2: coeficiente de determinación
    rmse: raíz del error cuadrático medio
    """
    # Construir la matriz de Vandermonde manualmente
    X = np.zeros((len(x), degree + 1))
    for i in range(degree + 1):
        X[:, i] = x**i
    
    # Resolver la ecuación de mínimos cuadrados: (X^T * X) * coeffs = X^T * y
    XtX = X.T @ X
    Xty = X.T @ y
    
    # Resolver el sistema de ecuaciones
    coeffs = np.linalg.solve(XtX, Xty)
    
    # Calcular predicciones
    y_pred = X @ coeffs
    
    # Calcular R² = 1 - SSres/SStot
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calcular RMSE
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    
    return coeffs, y_pred, r2, rmse

def predict_polynomial(coeffs, x_value):
    """
    Predice un valor y para un x específico usando coeficientes polinomiales.
    
    Parámetros:
    coeffs: coeficientes del polinomio (de menor a mayor grado)
    x_value: valor x para predecir
    
    Retorna:
    y_pred: valor y predicho
    """
    return sum(coef * (x_value**i) for i, coef in enumerate(coeffs))

def get_polynomial_equation(coeffs):
    """
    Genera la ecuación polinomial como string.
    
    Parámetros:
    coeffs: coeficientes del polinomio (de menor a mayor grado)
    
    Retorna:
    equation: string con la ecuación
    """
    equation = f"y = {coeffs[0]:.4f}"
    
    for i in range(1, len(coeffs)):
        if coeffs[i] >= 0:
            equation += f" + {coeffs[i]:.4f}x^{i}"
        else:
            equation += f" - {abs(coeffs[i]):.4f}x^{i}"
            
    return equation

# Ejemplo de datos
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 5, 4, 6, 8, 9, 10, 11, 13])

# Grado del polinomio a mostrar como principal
selected_degree = 2  # Puedes cambiar este valor

# Valor específico para predicción
x_specific = 5.5  # Valor para predecir

# Almacenar resultados para diferentes grados
max_degree = min(10, len(x) - 1)  # El grado máximo no puede exceder n-1
all_y_pred = {}
all_coeffs = {}
all_r2 = {}
all_rmse = {}
all_y_specific = {}

# Calcular regresiones para grados 1 a max_degree
for degree in range(1, max_degree + 1):
    coeffs, y_pred, r2, rmse = polynomial_regression_manual(x, y, degree)
    all_coeffs[degree] = coeffs
    all_y_pred[degree] = y_pred
    all_r2[degree] = r2
    all_rmse[degree] = rmse
    all_y_specific[degree] = predict_polynomial(coeffs, x_specific)

# Crear tabla comparativa de grados
comparison_data = {
    'Grado': list(range(1, max_degree + 1)),
    'R²': [all_r2[d] for d in range(1, max_degree + 1)],
    'RMSE': [all_rmse[d] for d in range(1, max_degree + 1)],
    f'Predicción en x={x_specific}': [all_y_specific[d] for d in range(1, max_degree + 1)]
}
df_comparison = pd.DataFrame(comparison_data)
print(f"\nComparación de diferentes grados para regresión polinomial:")
print(df_comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# Obtener resultado específico del grado seleccionado
coeffs = all_coeffs[selected_degree]
y_pred = all_y_pred[selected_degree]
r2 = all_r2[selected_degree]
rmse = all_rmse[selected_degree]
y_specific = all_y_specific[selected_degree]

# Calcular errores para el grado seleccionado
errors = y - y_pred
squared_errors = errors**2

# Crear tabla con resultados detallados para el grado seleccionado
table_data = {
    'x': x,
    'y (real)': y,
    'y (predicho)': y_pred,
    'Error': errors,
    'Error²': squared_errors
}
df = pd.DataFrame(table_data)
print(f"\nTabla de resultados de la regresión polinomial (Grado {selected_degree}):")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Generar ecuación del polinomio
equation = get_polynomial_equation(coeffs)

# Imprimir información sobre la regresión seleccionada
print(f"\nResultados de la regresión polinomial (Grado {selected_degree}):")
print(f"Ecuación: {equation}")
print(f"Coeficiente de determinación (R²): {r2:.6f}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse:.6f}")
print(f"Predicción para x = {x_specific}: y = {y_specific:.6f}")

# Crear puntos para la curva suave de regresión
x_curve = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
y_curves = {}

for degree in range(1, max_degree + 1):
    y_curves[degree] = np.array([predict_polynomial(all_coeffs[degree], xi) for xi in x_curve])

# Crear gráfica
plt.figure(figsize=(12, 7))

# Graficar puntos de datos
plt.scatter(x, y, color='blue', s=50, label='Datos reales', zorder=5)

# Graficar curva de regresión para el grado seleccionado
plt.plot(x_curve, y_curves[selected_degree], color='red', linewidth=2, 
         label=f'Regresión grado {selected_degree}: R² = {r2:.4f}')

# Graficar líneas de error
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'g--', alpha=0.5)

# Marcar el punto específico de predicción
plt.scatter(x_specific, y_specific, color='purple', s=100, marker='*', 
            label=f'Predicción en x={x_specific}: y={y_specific:.4f}')
plt.axvline(x=x_specific, color='purple', linestyle='--', alpha=0.3)
plt.axhline(y=y_specific, color='purple', linestyle='--', alpha=0.3)

plt.title(f'Regresión Polinomial (Grado {selected_degree}) por Mínimos Cuadrados')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Para analizar la calidad del ajuste: gráfica de residuos
plt.figure(figsize=(10, 4))
plt.scatter(x, errors, color='purple')
plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
plt.title(f'Gráfico de Residuos - Regresión Grado {selected_degree}')
plt.xlabel('x')
plt.ylabel('Residuos (y - ŷ)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Comparación de diferentes grados
plt.figure(figsize=(14, 8))
plt.scatter(x, y, color='blue', s=50, label='Datos reales', zorder=5)

colors = ['green', 'red', 'purple', 'orange', 'brown', 'magenta', 'cyan', 'olive', 'gray', 'pink']
for degree in range(1, min(len(colors) + 1, max_degree + 1)):
    plt.plot(x_curve, y_curves[degree], color=colors[degree-1], linestyle='-', 
             label=f'Grado {degree}: R² = {all_r2[degree]:.4f}')

plt.title('Comparación de Regresiones Polinomiales de Diferentes Grados')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()