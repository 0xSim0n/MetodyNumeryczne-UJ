import numpy as np
import matplotlib.pyplot as plt


def function_1(x):
    return 1/(1+50*(x**2))


def function_2(x):
    return 1/(1+5*(x**2))


def function_3(x):
    return 1/(1+(x**2))


def point_a(n):
    return [-1 + 2*i/n for i in range(n+1)]


def point_b(n):
    return np.cos([((2 * i + 1) / (2 * (n + 1))) * np.pi for i in range(n+1)])


def interpolation(function, point, arg, n):
    x = point(n)
    y = list(map(lambda a: function(a), x))

    new_y = []
    for a in arg:
        value = sum(y[i] * np.prod([(a - x[k]) / (x[i] - x[k]) for k in range(n+1) if i != k]) for i in range(n+1))
        new_y.append(value)

    return new_y


def plot(function, point, arg, title):
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(arg, function(arg), label='f(x)')
    for n in [2, 5, 9, 13, 15]:
        plt.plot(arg, interpolation(function, point, arg, n), label=f'W_{n}(x)')
    plt.grid()
    plt.legend()
    plt.show()


new = np.arange(-1.0, 1.01, 0.01)

plot(function_1, point_a, new, 'Wykres dla funkcji: f(x)=1/(1+50x^2)\n siatka: x_i = -1+2(i/n)')
plot(function_1, point_b, new, 'Wykres dla funkcji: f(x)=1/(1+50x^2)\n siatka: x_i= cos((2i+1)/(2(n+1))*Pi)')
plot(function_2, point_a, new, 'Wykres dla funkcji: f(x)=1/(1+5x^2)\n siatka: x_i = -1+2(i/(n))')
plot(function_2, point_b, new, 'Wykres dla funkcji: f(x)=1/(1+5x^2)\n siatka: x_i= cos((2i+1)/(2(n+1))*Pi)')
plot(function_3, point_a, new, 'Wykres dla funkcji: f(x)=1/(1+x^2)\n siatka: x_i = -1+2(i/(n))')
plot(function_3, point_b, new, 'Wykres dla funkcji: f(x)=1/(1+x^2)\n siatka: x_i= cos((2i+1)/(2(n+1))*Pi)')
