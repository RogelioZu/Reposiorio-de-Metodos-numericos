import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función dada
def f(x):
    return (x * np.exp(x))

# Derivada centrada de orden 2
def derivada_centrada(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

# Extrapolación de Richardson
def richardson_derivada(f, x0, h_inicial, niveles):
    hs = [h_inicial / (2**i) for i in range(niveles)]
    filas = []
    
    D = [[0 for _ in range(niveles)] for _ in range(niveles)]
    
    for i, h in enumerate(hs):
        x_menos = x0 - h
        x_mas = x0 + h
        fx_menos = f(x_menos)
        fx_mas = f(x_mas)
        
        D[i][0] = derivada_centrada(f, x0, h)
        
        for j in range(1, i + 1):
            D[i][j] = D[i][j-1] + (D[i][j-1] - D[i-1][j-1]) / (4**j - 1)
        
        fila = {
            "h": round(h, 8),
            "x0 - h": round(x_menos, 8),
            "x0 + h": round(x_mas, 8),
            "f(x0 - h)": round(fx_menos, 6),
            "f(x0 + h)": round(fx_mas, 6)
        }
        for j in range(i + 1):
            fila[f"N{j+1}(h)"] = round(D[i][j], 6)
        
        filas.append(fila)

    columnas = ["h", "x0 - h", "x0 + h", "f(x0 - h)", "f(x0 + h)"] + [f"N{j+1}(h)" for j in range(niveles)]
    tabla = pd.DataFrame(filas, columns=columnas)

    return tabla, D

# Ejecutar algoritmo
x0 = 2
h_inicial = 0.2
niveles = 10

tabla, D = richardson_derivada(f, x0, h_inicial, niveles)
print(tabla)

# Gráfica de convergencia
plt.figure(figsize=(10, 5))
for j in range(niveles):
    y_vals = [D[i][j] for i in range(j, niveles)]
    h_vals = [h_inicial / (2**i) for i in range(j, niveles)]
    plt.plot(h_vals, y_vals, marker='o', label=f"N{j+1}(h)")

plt.xscale('log')
plt.xlabel("h (log scale)")
plt.ylabel("Aproximación de la derivada")
plt.title("Convergencia - Extrapolación de Richardson")
plt.grid(True)
plt.legend()
plt.show()
