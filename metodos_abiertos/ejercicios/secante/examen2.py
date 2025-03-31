from math import *
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


# Función a evaluar
def f(x):
    return x**3 - 5*x + np.exp(-x) -2

# Algoritmo de la secante
def secante(f, x0, x1 ,imax ,tol):

    resultados = []
    n = 0  # Contador de iteraciones
    cumple = False
    xanterior = x1
    

    while (not cumple and n < imax):
        x = x1 - ( ( f(x1) * (x1 - x0) ) / ( f(x1) - f(x0)) ) 

        # Calcular el criterio de convergencia
        ccn = abs((x - xanterior) / x)  # Error relativo
        
        # Imprimir iteración actual con alineación adecuada
        if n == 0:
            x = x0
            xr = x1
            fxr = f(xr)
            fxn = f(x)
            resultados.append([n,x,fxn,"----"])
            resultados.append([n,xr,fxr,ccn])
            
           
       
        else:
            resultados.append([n,x,f(x), ccn])
            

        if ccn < tol:
            print("solucion aproximada: ")
            cumple = True
        
        xanterior = x
        x0 = x1
        x1 = x
        n += 1

    if n < imax:
        return x, resultados, n
    else: 
        raise ValueError("La funcion no converge")


def main():
    # Valores iniciales
    x0 = 0
    x1 = 2
    iteraciones = 30
    tol = 5e-11
    # Llamada al algoritmo
    raiz, resultados, iter = secante(f, x0, x1, iteraciones, tol)

    #Imprimimos la tabla
    headers = ["Iteraciones", "Xn", "f(Xn)", "CCn"]
    print(tabulate(resultados, headers=headers, 
                  floatfmt=".10f", 
                  tablefmt="fancy_grid", 
                  numalign="right",
                  stralign="center"))
    
    print(f'valor de la funcion en el punto: f({raiz:.30f}) = {f(raiz):.30f}')
    print(f"numero de iteraciones: {iter}")
    print(f"valor de x en la ultima iteracion: {raiz:.30f}")

    # Graficar la función
    f2 = np.vectorize(f)
    x = np.linspace(x0, x1, 100)
    y = f2(x)

    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.scatter(raiz, f(raiz), color='red', zorder=3, label=f'Raíz ≈ {raiz:.5f}')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__": main()
