import math

def f(x):
    return math.sqrt(1 - x**3)



def regla_trapecio(a,b,n,f):
    cont = 0
    aprox = 0
    h = (b - a) / n
    if n == 1:
        aprox = ((b - a)/ 2) * (f(a) + f (b))
        

    elif n > 1:
        suma_interna = 0
        print("|    |  x  |  y  |")
        for i in range(1, n): # Suma los puntos interiores
            cont += 1
            x_i = a + i * h
            suma_interna += f(x_i)

            print(f"x{cont} | {x_i} | {f(x_i)} |")
            
        aprox = (h /2) * (f(a) + f(b) + 2 *(suma_interna))

    else: # n debe ser al menos 1
        raise ValueError("El número de subintervalos 'n' debe ser 1 o mayor.")

    return aprox


if __name__ == '__main__':
    print(f"aproximacion de la funcion {regla_trapecio(0, 1, 1, f)}")