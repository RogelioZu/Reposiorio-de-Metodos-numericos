import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ajustar_regresion_exponencial(x, y, grado):
    """Ajusta regresión exponencial: ln(y) ≈ polinomio(x)"""
    ln_y = np.log(y)
    X = np.vander(x, grado + 1, increasing=True)
    coef = np.linalg.lstsq(X, ln_y, rcond=None)[0]
    return coef

def evaluar_polynomial(coef, x):
    return sum(c * x**i for i, c in enumerate(coef))

def imprimir_polinomio(coef, variable='x'):
    """Devuelve una cadena con el polinomio en forma legible"""
    términos = []
    for i, c in enumerate(coef):
        término = f"{c:.4f}"
        if i == 1:
            término += f" * {variable}"
        elif i > 1:
            término += f" * {variable}^{i}"
        términos.append(término)
    return " + ".join(términos)

def comparar_regresiones_exponenciales(x, y, x_eval, max_grado=3):
    resultados = []
    x_vals = np.linspace(min(x), max(x), 200)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', label='Datos originales')

    for grado in range(1, max_grado + 1):
        coef = ajustar_regresion_exponencial(x, y, grado)

        # Evaluación en el punto x_eval
        ln_y_eval = evaluar_polynomial(coef, x_eval)
        y_aprox = np.exp(ln_y_eval)
        resultados.append([grado, y_aprox])

        # Imprimir polinomio
        ln_polinomio = imprimir_polinomio(coef, variable='x')
        print(f"Grado {grado} → ln(y) = {ln_polinomio}")
        print(f"          → y = exp({ln_polinomio})\n")

        # Graficar curva ajustada
        y_graf = [np.exp(evaluar_polynomial(coef, xi)) for xi in x_vals]
        plt.plot(x_vals, y_graf, label=f'Grado {grado}')

    plt.axvline(x=x_eval, color='gray', linestyle='--', label=f'x = {x_eval}')
    plt.title('Regresiones Exponenciales (sin sklearn)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Tabla de resultados
    tabla = pd.DataFrame(resultados, columns=["Grado", f"Aproximación en x={x_eval}"])
    print("Tabla de aproximaciones:")
    print(tabla)

# =======================
# EJEMPLO DE USO
# =======================

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2.5, 3.5, 5.0, 6.8, 9.5, 13.0, 18.0, 24.0])
x_a_evaluar = 3.5

comparar_regresiones_exponenciales(x, y, x_eval=x_a_evaluar, max_grado=3)
