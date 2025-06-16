import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from scipy.interpolate import lagrange
from scipy.integrate import simpson, quad

#Funcion a integrar
def f (x):
    return (1 + x**3)**(1/3)


def regla_simpson13(f,a,b,n):
    h = (b - a) / n
    xs = np.linspace(a,b, n + 1)
    ys = f(xs)
    print(f" h = {h}")
    print("Tabulación:")
    print("x\t\ty")
    print("-" * 20)
    for i in range(len(xs)):
        print(f"{xs[i]:.3f}\t\t{ys[i]:.6f}")
    
    r = h * (ys[0] + 4*sum(ys[1:n:2]) + 2*sum(ys[2:n-1:2]) + ys[n])/3
    return r

def grafica_simpson(f,a,b,n):
    x = np.linspace(a,b)
    y = f(x)
    fig, ax = plt.subplots()
    ax.plot(x,y, 'b',linewidth=1.7)
    ax.set_ylim(bottom=0)

    h = (b - a)/n
    x0,x1,x2 = a, a+h, a+2*h
    i = 0
    patterns =( ' - ' , '/ ' , ' \\ ' , 'O ' , '. ' , 'o ' , '* ' , ' \\ ' , '/ ' , ' - ' , 'x ' , '+ ')
    for i in range(0,n,2):
        xx = np.array([x0,x1,x2])
        yy = np.array([f(x0), f(x1), f(x2)])
        pol = lagrange(xx, yy)

       
        ix = np.linspace ( x0 , x1 )
        iy = pol ( ix )
        verts = [( x0 , 0) , * zip ( ix , iy ) ,( x1 , 0) ]
        poly = Polygon(verts, facecolor = '0.9', edgecolor='0.5', hatch= patterns[i])
        ax.add_patch(poly)

        ix = np.linspace(x1,x2)
        iy = pol(ix)
        verts = [(x1,0), *zip(ix,iy), (x2,0)]
        poly = Polygon(verts, facecolor = '0.9', edgecolor='0.5', hatch= patterns[i])
        ax.add_patch(poly)
        ax.add_patch(poly)

        x0 , x1 , x2 = x2 , x2 +h , x2 +2* h
    plt . title ( ' Regla de Simpson 1/3 ')
    # fig . savefig (" int_simpson13C . pdf " , bbox_inches = ' tight ')
    return plt

if __name__ == '__main__':
    a = 0
    b = 1
    n= 6
    area = regla_simpson13(f,a,b,n)


    valor_real, error_quad = quad(f, a, b)
    print(f'Valor real (quad):        {valor_real:.6f}')
     # Calcular error absoluto y relativo
    error_absoluto = abs(area - valor_real)
    error_relativo = (error_absoluto / abs(valor_real)) * 100
    print(f'\nError absoluto: {error_absoluto:.6f}')
    print(f'Error relativo: {error_relativo:.4f}%')
    print('Integral:', area)
    g = grafica_simpson(f,a,b,n)
    g.show()
