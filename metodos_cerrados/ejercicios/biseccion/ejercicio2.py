import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from numpy import exp

def f(x):
    """Función objetivo a encontrar su raíz."""
    return exp(x) - 1 - x - ((x**2)/2)

def biseccion(a, b, tol, max_iter):
   
    xi = a
    xu = b
    xr_anterior = 0
    
    # Listas para almacenar los valores para la gráfica
    xr_points = []
    yr_points = []
    
    # Lista para almacenar los datos de cada iteración para tabulate
    tabla_datos = []
    
    # Verificar si hay una raíz en el intervalo [a, b]
    if f(xi) * f(xu) > 0:
        return None, 0, [["Error", "No hay raíz en el intervalo dado, f(a) y f(b) tienen el mismo signo"]]
    
    for i in range(max_iter):
        xr = (xi + xu) / 2
        f_xr = f(xr)
        
        # Calcular error relativo (excepto en primera iteración)
        error_rel = abs((xr - xr_anterior) / xr) if i > 0 else float('inf')
        xr_anterior = xr
        
        # Guardar puntos para la gráfica
        xr_points.append(xr)
        yr_points.append(f_xr)
        
        # Determinar qué valores guardar según el signo del producto
        if f(xi) * f_xr < 0:
            xder = xu
            xu = xr
            fila = [i, xi, xr, xder, f_xr]
        else:
            xizq = xi
            xi = xr
            fila = [i, xizq, xr, xu, f_xr]
        
        # Añadir error relativo a la fila
        if i > 0:
            fila.append(error_rel)
        else:
            fila.append(None)  # No hay error relativo en la primera iteración
        
        # Añadir la fila a la tabla
        tabla_datos.append(fila)
        
        # Criterio de parada: error relativo menor que la tolerancia
        if error_rel < tol and i > 0:
            break
    
    # Generar la visualización
    generar_grafica(a, b, xr_points, yr_points)
    
    return xr, i, tabla_datos

def generar_grafica(a, b, xr_points, yr_points):
    """Genera una gráfica del proceso de bisección."""
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 12})
    
    # Generar puntos para graficar la función
    x = np.linspace(a-0.5, b+0.5, 1000)
    y = [f(val) for val in x]
    
    # Graficar la función y los puntos de bisección
    plt.plot(x, y, 'b-', label='f(x)', linewidth=2.5, alpha=0.7)
    plt.plot(xr_points, yr_points, 'ro-', label='Puntos de bisección', markersize=8, linewidth=1.5, alpha=0.6)
    
    # Destacar la raíz encontrada
    if xr_points:
        plt.plot(xr_points[-1], yr_points[-1], 'go', label='Raíz encontrada', markersize=12)
    
    # Configuraciones adicionales de la gráfica
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

def ejecutar_biseccion(a, b, tol, max_iter):
    """Ejecuta el método de bisección y muestra los resultados."""
    print(f"\nBuscando raíz en el intervalo [{a}, {b}] con tolerancia {tol}")
    print("Ejecutando método de bisección...\n")
    
    xr, iter_count, tabla_datos = biseccion(a, b, tol, max_iter)
    
    if xr is None:
        print(tabulate(tabla_datos, headers=["Error"], tablefmt="fancy_grid"))
        return
    
    # Definir encabezados para la tabla
    headers = ["Iteración", "Xi", "Xr", "Xu", "f(Xr)"]
    if len(tabla_datos) > 1:  # Si hay más de una iteración
        headers.append("Error Relativo")
    
    # Preparar los datos para mostrar (sin error relativo en primera iteración)
    tabla_mostrar = []
    for i, fila in enumerate(tabla_datos):
        if i == 0:  # Primera iteración - omitir el error relativo
            tabla_mostrar.append(fila[:-1])  # Excluir el último elemento (None)
        else:
            tabla_mostrar.append(fila)
    
    # Mostrar la tabla de iteraciones formateada
    print(tabulate(tabla_mostrar, headers=headers, 
                   tablefmt="fancy_grid", 
                   floatfmt=".8f",
                   numalign="right"))
    
    print(f"\nResumen:")
    print(f"  * Raíz aproximada: {xr:.10f}")
    print(f"  * Valor de f(x) en la raíz: {f(xr):.10e}")
    print(f"  * Número de iteraciones: {iter_count + 1}")
    print(f"  * Error relativo final: {tabla_datos[iter_count][5]}")
    
    plt.show()

if __name__ == '__main__':
    ejecutar_biseccion(0, 0.5, 5e-11, 50)