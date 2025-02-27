import numpy as np
import matplotlib.pyplot as plt
from math import *
from tabulate import tabulate
import sympy as sp

def f(x):
    return x**3 + 2*x**2 + 10*x -20

def derivada(funcion_original):

    x = sp.Symbol('x')

    expresion = funcion_original(x)

    derivada = sp.diff(expresion, x)

    funcion_derivada = sp.lambdify(x, derivada)
    
    return funcion_derivada

def newton_raphson(f, x0, tol=1e-6, max_iter=100):

    df = derivada(f)
    x = x0
    iteraciones = 0
    historico = [x0]
    
    tabla_datos = []
    
    while iteraciones < max_iter:
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
        
        x_nuevo = x - f_valor / df_valor
        
        historico.append(x_nuevo)
        
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
    
    x0 = 2
    
    raiz, iteraciones, historico, tabla_datos = newton_raphson(f, x0)
    
    headers = ["Iteración", "Valor de x", "f(x)", "f'(x)", "Error |xᵢ - xᵢ₋₁|"]
    print("\nTabla de iteraciones del método de Newton-Raphson:")
    print(tabulate(tabla_datos, headers=headers, 
                  floatfmt=".10f", 
                  tablefmt="fancy_grid", 
                  numalign="right",
                  stralign="center"))
    
    print(f"\nRaíz encontrada: {raiz:.10f}")
    print(f"Valor de f en la raíz: {f(raiz):.10e}")
    print(f"Número de iteraciones: {iteraciones}")
    
    x_vals = np.linspace(raiz-3, raiz+3, 1000)
    y_vals = [f(xi) for xi in x_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x) = e^x - 3x²')
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