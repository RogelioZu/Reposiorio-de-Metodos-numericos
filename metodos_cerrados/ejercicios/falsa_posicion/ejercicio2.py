#from math import exp
#from math import atan
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def f(x):
    return np.atan(x) - (2*x / (1 + x**2))
    #x**3 + 2*x**2 + 10*x -20
    #atan(x) - (2*x / (1 + x**2))


def falsa_posicion(a,b,tol,iter):
    xi = a
    xu = b
    
    xr_anterior = 0  # Para calcular el error relativo
    
    # Listas para almacenar los valores para la gráfica
    xr_points = []
    yr_points = []
    
    # Lista para almacenar los datos de la tabla
    tabla_datos = []
    
    if (f(xi) * f(xu) <= 0):
        for i in range(iter):
            yn_1 = f(xi)
            yn = f(xu)
            
            xr = xu - (f(xu)*(xi-xu))/(f(xi) - f(xu))

            # Calcular error relativo (excepto en primera iteración)
            error_rel = abs((xr - xr_anterior)/xr) if i > 0 else float('inf')
            xr_anterior = xr  # Guardar xr actual para siguiente iteración
            
            # Guardar puntos para la gráfica
            xr_points.append(xr)
            yr_points.append(f(xr))
                
            # Cambiamos el criterio de parada para usar el error relativo
            if error_rel < tol and i > 0:
                print(f"Resultado aproximado: {xr}, con un numero de iteraciones {i}")
                break
                
            if f(xi) * f(xr) < 0:
                xder = xu
                xu = xr
                
                # Preparar datos para la tabla
                if i == 0:
                    fila = [i, xi, xr, xder, yn_1, yn, f(xr), "---"]
                else:
                    fila = [i, xi, xr, xder, yn_1, yn, f(xr), error_rel]
                tabla_datos.append(fila)
            else:
                xizq = xi
                xi = xr
                
                # Preparar datos para la tabla
                if i == 0:
                    fila = [i, xizq, xr, xu, yn_1, yn, f(xr), "---"]
                else:
                    fila = [i, xizq, xr, xu, yn_1, yn, f(xr), error_rel]
                tabla_datos.append(fila)
        
        # Mostrar la tabla con los resultados
        headers = ["Iteración", "Xi", "Xn", "Xu", "Y n-1", "Y n", "f(Xn)", "Error Rel."]
        
        # Preparar los datos para mostrar (sin error relativo en primera iteración)
        tabla_mostrar = []
        for i, fila in enumerate(tabla_datos):
            if i == 0:  # Primera iteración - mantener la columna vacía
                tabla_mostrar.append(fila)
            else:
                tabla_mostrar.append(fila)
        
        print("\nTabla de iteraciones del método de Falsa Posición:")
        print(tabulate(tabla_mostrar, headers=headers, 
                      floatfmt=".8f", 
                      tablefmt="fancy_grid", 
                      numalign="right",
                      stralign="center"))
        
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({'font.size': 12})
        x = np.linspace(a-0.5, b+0.5, 1000)
        y = [f(val) for val in x]
        plt.plot(x, y, 'b-', label='f(x)', linewidth=2.5, alpha=0.7)
        plt.plot(xr_points, yr_points, 'ro-', label='Puntos de falsa posición', markersize=8, linewidth=1.5, alpha=0.6)
        plt.plot(xr_points[-1], yr_points[-1], 'go', label='Raíz encontrada', markersize=12)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=1.5)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.xlim(a-0.5, b+0.5)
        ymin, ymax = min(y), max(y)
        plt.ylim(ymin-0.5, ymax+0.5)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc='best', fontsize=10, facecolor='white', edgecolor='gray', framealpha=0.9)
        plt.tight_layout()
        print(f"\nResumen:")
        print(f"  * Raíz aproximada: {xr:.30f}")
        print(f"  * Valor de f(x) en la raíz: {f(xr):.10e}")
        print(f"  * Número de iteraciones: {i + 1}")
        print(f"  * Error relativo final: {tabla_datos[i][5]:.10e}")
        plt.show()

if __name__ == '__main__':
  
    falsa_posicion(0.1, 1.5, 5e-18, 50)