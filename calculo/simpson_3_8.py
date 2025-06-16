import numpy as np
from scipy . interpolate import lagrange
import matplotlib . pyplot as plt
from matplotlib . patches import Polygon


def f(x):
    return x

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


if __name__ == '__main__':
    a = 2
    b = 8
    n = 3
    area = regla_simpson38(f,a,b,n)
    print('Integral = ', area)
    