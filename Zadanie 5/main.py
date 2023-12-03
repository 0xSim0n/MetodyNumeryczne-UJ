import numpy as np
import copy
import random
import matplotlib.pyplot as plt

def jacobi_method(x, stop):
    result = []
    norm = 0
    while stop:
        y = x.copy()
        for i in range(n):
            if i == 0:
                x[i] = (b[i] - y[i+1] - 0.15*y[i+2]) / 3
            elif i == 1:
                x[i] = (b[i] - y[i-1] - y[i+1] - 0.15*y[i+2]) / 3
            elif i == n-2:
                x[i] = (b[i] - y[i-1] - 0.15*y[i-2] - y[i+1]) / 3
            elif i == n-1:
                x[i] = (b[i] - y[i-1] - 0.15*y[i-2]) / 3
            else:
                x[i] = (b[i] - y[i-1] - 0.15*y[i-2] - y[i+1] - 0.15*y[i+2]) / 3
        norm1 = np.sqrt(sum((a - b)**2 for a, b in zip(x, y)))
        result.append(copy.deepcopy(x))

        if abs(norm - norm1) < 1e-12:
            break
        norm = norm1
        stop = stop - 1

    print("Result obtained by the Jacobi method:")
    print(x)
    print()
    return result

def gauss_seidel(x, stop):
    result = []
    norm = 0
    while stop:
        y = x.copy()
        for i in range(n):
            if i == 0:
                x[i] = (b[i] - x[i+1] - 0.15*x[i+2]) / 3
            elif i == 1:
                x[i] = (b[i] - x[i-1] - y[i+1] - 0.15*y[i+2]) / 3
            elif i == n-2:
                x[i] = (b[i] - x[i-1] - 0.15*x[i-2] - y[i+1]) / 3
            elif i == n-1:
                x[i] = (b[i] - x[i-1] - 0.15*x[i-2]) / 3
            else:
                x[i] = (b[i] - x[i-1] - 0.15*x[i-2] - y[i+1] - 0.15*y[i+2]) / 3

        norm1 = np.sqrt(sum((a - b)**2 for a, b in zip(x, y)))
        result.append(copy.deepcopy(x))

        if abs(norm - norm1) < 1e-12:
            break
        norm = norm1
        stop = stop - 1

    print("Result obtained by the Gauss-Seidel method:")
    print(x)
    print()
    return result

n = 124
stop = 300
x = random.sample(range(300), 124)
b = list(range(1, n + 1))

result1 = jacobi_method(x.copy(), stop)
result2 = gauss_seidel(x.copy(), stop)

w1 = [np.sqrt(sum((a - b)**2 for a, b in zip(e, result1[-1]))) for e in result1[:-1]]
w2 = [np.sqrt(sum((a - b)**2 for a, b in zip(e, result2[-1]))) for e in result2[:-1]]

def plot_comparison(w1, w2):
    plt.grid(True)
    plt.xlabel('n')
    plt.ylabel("$|x(n) - x(last)|$")
    plt.yscale('log')
    plt.plot(range(1, len(w1) + 1), w1)
    plt.plot(range(1, len(w2) + 1), w2)
    plt.legend(['Jacobi Method', 'Gauss-Seidel Method'])
    plt.title('Comparison of Iterative Methods')
    plt.show()

plot_comparison(w1, w2)
