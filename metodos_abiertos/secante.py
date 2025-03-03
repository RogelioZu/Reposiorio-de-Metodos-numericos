from math import *
import numpy as np
import matplotlib.pyplot as plt



# Función a evaluar
def f(x):
    return x**3 + 2*x**2 + 10*x -20

# Algoritmo de la secante
def secante(f, x0, x1 ,imax ,tol):

    n = 0  # Contador de iteraciones
    cumple = False
    xanterior = x1

    # Encabezado con mejor alineación
    print(f'{"n":<5}  {"Xn":<15} {"f(Xn)":<15} {"CCn":<15}')
    print('-' * 80)

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
            print(f'{n:<5}  {x:<15.10f}  {fxn:<15.10f} {"----":<15}')
            print(f'{n:<5}  {xr:<15.10f}  {fxr:<15.10f} {ccn:<15}')
           
       
        else:
            print(f'{n :<5}  {x:<15.10f}  {f(x):<15.10f} {ccn:<15.10f}')

        if ccn < tol:
            print("solucion aproximada: ")
            cumple = True
        
        xanterior = x
        x0 = x1
        x1 = x
        n += 1

    if n < imax:
        return x
    else: 
        raise ValueError("La funcion no converge")


def main():
    # Valores iniciales
    x0 = 1
    x1 = 2
    iteraciones = 30
    tol = 5e-11

    # Llamada al algoritmo
    raiz = secante(f, x0, x1, iteraciones, tol)
    print(f'f({raiz:.5f}) = {f(raiz):.5f}')

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
