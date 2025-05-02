import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def neville(x, y, valor):
    n = len(x)
    P = np.zeros((n, n))
    P[:, 0] = y  # Primera columna con f(x_i)

    # Construcción del esquema de Neville
    for j in range(1, n):
        for i in range(n - j):
            P[i][j] = ((valor - x[i + j]) * P[i][j - 1] + (x[i] - valor) * P[i + 1][j - 1]) / (x[i] - x[i + j])

    return P[0, n - 1], P

# Para graficar el polinomio, usaremos interpolación en muchos puntos
def neville_polynomial(x_data, y_data, x_eval):
    return [neville(x_data, y_data, xi)[0] for xi in x_eval]

# Datos de ejemplo
x = np.array([0, 1, 2,3 ])
y = np.array([2.0, 2.5, 3, 2.8])
valor_a_interpolar = 1.5

# Interpolación
resultado, tabla_neville = neville(x, y, valor_a_interpolar)

# Mostrar tabla como DataFrame
columnas = [f'P_{i}' for i in range(len(x))]
tabla_df = pd.DataFrame(tabla_neville, columns=columnas)
print("Tabla del esquema de Neville:")
print(tabla_df)
print(f"\nEl valor interpolado en x = {valor_a_interpolar} es: {resultado:.4f}")

# Gráfica
x_graf = np.linspace(min(x) - 1, max(x) + 1, 500)
y_graf = neville_polynomial(x, y, x_graf)

plt.figure(figsize=(8, 5))
plt.plot(x_graf, y_graf, label="Polinomio interpolante (Neville)", color='blue')
plt.scatter(x, y, color='red', label="Puntos dados")
plt.scatter(valor_a_interpolar, resultado, color='green', label=f"Interpolado ({valor_a_interpolar:.2f}, {resultado:.2f})")
plt.title("Interpolación de Neville")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()