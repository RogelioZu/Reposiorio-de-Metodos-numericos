import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp

# Definimos la expresión simbólica
x_sym = sp.Symbol('x')
expresion = sp.log(x_sym) - (x_sym -2)**2

# Convertimos la expresión en una función numérica
def f(x):
    return float(expresion.subs(x_sym, x))

# Función para calcular la derivada de la función
def derivada():
    derivada_simbolica = sp.diff(expresion, x_sym)
    funcion_derivada = sp.lambdify(x_sym, derivada_simbolica)
    return funcion_derivada

# Función del método, que recibe un punto inicial, tolerancia y número de máximas iteraciones
def newton_raphson(f, x0, tol=5e-9, max_iter=100):
    # Aquí se calcula la derivada
    df = derivada()
    x = x0
    iteraciones = 0
    # Esta variable va guardando los valores que va tomando x
    historico = [x0]
    tabla_datos = []
    while iteraciones < max_iter:
        # Evaluamos en cada iteración x en la función y su derivada
        f_valor = f(x)
        df_valor = df(x)
        if iteraciones == 0:
            tabla_datos.append([iteraciones, x, f_valor, df_valor, "---"])
        else:
            error = abs(x - historico[-2]) if len(historico) > 1 else "---"
            tabla_datos.append([iteraciones, x, f_valor, df_valor, error])
        if abs(df_valor) < 1e-10:
            print("Advertencia: Derivada cercana a cero. Posible divergencia.")
            break
        # Aquí se aplica la fórmula del método
        x_nuevo = x - f_valor / df_valor
        historico.append(x_nuevo)
        # Si llegamos a la tolerancia entonces paramos
        if abs(x_nuevo - x) < tol:
            error = abs(x_nuevo - x)
            tabla_datos.append([iteraciones + 1, x_nuevo, f(x_nuevo), df(x_nuevo), error])
            x = x_nuevo
            iteraciones += 1
            break
        x = x_nuevo
        iteraciones += 1
    return x, iteraciones, historico, tabla_datos

if __name__ == "__main__":
    x0 = 1.5
    raiz, iteraciones, historico, tabla_datos = newton_raphson(f, x0)
    
    # Aquí se hace la tabulación usando la librería tabulate
    headers = ["Iteración", "Valor de x", "f(x)", "f'(x)", "Error |xᵢ - xᵢ₋₁|"]
    print("\nTabla de iteraciones del método de Newton-Raphson:")
    print(tabulate(tabla_datos, headers=headers,
        floatfmt=".10f",
        tablefmt="fancy_grid",
        numalign="right",
        stralign="center"))
    
    print(f"\nRaíz encontrada: {raiz:.30f}")
    print(f"Valor de f en la raíz: {f(raiz):.10e}")
    print(f"Número de iteraciones: {iteraciones}")
    
    x_vals = np.linspace(raiz-3, raiz+3, 1000)
    y_vals = [f(xi) for xi in x_vals]
    
    # La graficación con matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(alpha=0.3)
    plt.scatter([raiz], [0], color='red', s=100, zorder=5, label=f'Raíz x ≈ {raiz:.6f}')
    
    for i, xi in enumerate(historico):
        plt.scatter([xi], [f(xi)], color='green', alpha=0.6)
        if i < len(historico) - 1:
            plt.plot([xi, xi, historico[i+1]], [f(xi), 0, 0], 'g--', alpha=0.4)
    
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()