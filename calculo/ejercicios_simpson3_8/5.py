import numpy as np
from scipy . interpolate import lagrange
import matplotlib . pyplot as plt
from matplotlib . patches import Polygon
from scipy.integrate import simpson, quad

def f(x):
    return (9 - x**2)**(1/3)

def regla_simpson38(f,a,b,n):
    h = (b - a) / n
    xs = np.linspace(a,b , n + 1)
    ys = f(xs)
     
    print("Tabulación Simpson 3/8:")
    print("x\t\ty")
    print("-" * 20)
    for i in range(len(xs)):
        print(f"{xs[i]:.3f}\t\t{ys[i]:.6f}")
    
    r = 3*h * (ys[0] + 3*sum(ys[1:n-1:3]) + 3*sum(ys[2:n:3]) + 2*sum(ys[3:n-2:3]) + ys[n]) / 8
    return r

def grafica_trapecios (f ,a ,b , n ):
    x = np . linspace (a , b )
    y = f(x)
    fig , ax = plt . subplots ()
    ax . plot (x , y , 'b ' , linewidth =1.7)
    ax . set_ylim ( bottom =0)
    h =( b - a ) / n
    x0 , x1 , x2 , x3 =a , a +h , a +2* h , a +3* h

    for i in range (0 ,n ,3) :
        xx = np . array ([ x0 , x1 , x2 , x3 ])
        yy = np . array ([ f ( x0 ) ,f ( x1 ) ,f ( x2 ) ,f ( x3 ) ])
        pol = lagrange ( xx , yy )
        ix = np . linspace ( x0 , x1 )
        iy = pol ( ix )
        verts = [( x0 , 0) , * zip ( ix , iy ) ,( x1 , 0) ]
        poly = Polygon ( verts , facecolor = ' 0.9 ' , edgecolor = ' 0.5 ')
        ax . add_patch ( poly )
        ix = np . linspace ( x1 , x2 )
        iy = pol ( ix )
        verts = [( x1 , 0) , * zip ( ix , iy ) ,( x2 , 0) ]
        poly = Polygon ( verts , facecolor = ' 0.9 ' , edgecolor = ' 0.5 ')
        ax . add_patch ( poly )
        ix = np . linspace ( x2 , x3 )
        iy = pol ( ix )
        verts = [( x2 , 0) , * zip ( ix , iy ) ,( x3 , 0) ]
        poly = Polygon ( verts , facecolor = ' 0.9 ' , edgecolor = ' 0.5 ')
        ax . add_patch ( poly )
        x0 , x1 , x2 , x3 = x3 , x3 +h , x3 +2* h , x3 +3* h
    plt . title ( ' Regla de Simpson 3/8 ')
    return plt

if __name__ == '__main__':
    a = 0
    b = 1
    n = 9
    area = regla_simpson38(f,a,b,n)

    valor_real, error_quad = quad(f, a, b)
    print(f'Valor real (quad):        {valor_real:.6f}')
     # Calcular error absoluto y relativo
    error_absoluto = abs(area - valor_real)
    error_relativo = (error_absoluto / abs(valor_real)) * 100
    print(f'\nError absoluto: {error_absoluto:.6f}')
    print(f'Error relativo: {error_relativo:.4f}%')
    print('Integral calculada = ', area)
    g = grafica_trapecios(f,a,b,n)
    g.show()
    