#from math import exp
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 + 2*x**2 + 10*x -20

def biseccion(a,b,tol,iter):
    xi = a
    xu = b
    xr_anterior = 0  # Para calcular el error relativo
    
    # Listas para almacenar los valores para la gráfica
    xr_points = []
    yr_points = []
    
    if (f(xi) * f(xu) <= 0):
        for i in range(iter):
            xr = (xi + xu)/2
            
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
                print("-----------------------------------------------------------------------------------")
                # Solo mostramos el error relativo después de la primera iteración
                if i == 0:
                    print(f"iteracion: {i} | valor Xizq: {xi:.8f} | Xn: {xr:.8f} | Valor Xder: {xder:.8f} | f(xn): {f(xr):.8f}")
                else:
                    print(f"iteracion: {i} | valor Xizq: {xi:.8f} | Xn: {xr:.8f} | Valor Xder: {xder:.8f} | f(xn): {f(xr):.8f} | Error relativo: {error_rel:.8f}")
            else:
                xizq = xi
                xi = xr
                print("-----------------------------------------------------------------------------------")
                # Solo mostramos el error relativo después de la primera iteración
                if i == 0:
                    print(f"iteracion: {i} | valor Xizq: {xizq:.8f} | Xn: {xr:.8f} | Valor Xder: {xu:.8f} | f(xn): {f(xr):.8f}")
                else:
                    print(f"iteracion: {i} | valor Xizq: {xizq:.8f} | Xn: {xr:.8f} | Valor Xder: {xu:.8f} | f(xn): {f(xr):.8f} | Error relativo: {error_rel:.8f}")
        
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({'font.size': 12})
        x = np.linspace(a-0.5, b+0.5, 1000)
        y = [f(val) for val in x]
        plt.plot(x, y, 'b-', label='f(x)', linewidth=2.5, alpha=0.7)
        plt.plot(xr_points, yr_points, 'ro-', label='Puntos de bisección', markersize=8, linewidth=1.5, alpha=0.6)
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
        plt.show()

if __name__ == '__main__':
    biseccion(0, 1, 5e-11, 50)