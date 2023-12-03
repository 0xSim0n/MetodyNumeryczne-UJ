import time
import matplotlib.pyplot as plt


def calculation(n):
    matrix = []
    matrix.append([0] + [0.2] * (n - 1))
    matrix.append([1.2] * n)
    matrix.append([0.1 / i for i in range(1, n)] + [0])
    matrix.append([0.15 / i ** 2 for i in range(1, n - 1)] + [0] + [0])

    x = list(range(1, n + 1))

    for i in range(1, n - 2):
        matrix[0][i] = matrix[0][i] / matrix[1][i - 1]
        matrix[1][i] = matrix[1][i] - matrix[0][i] * matrix[2][i - 1]
        matrix[2][i] = matrix[2][i] - matrix[0][i] * matrix[3][i - 1]

    matrix[0][n - 2] = matrix[0][n - 2] / matrix[1][n - 3]
    matrix[1][n - 2] = matrix[1][n - 2] - matrix[0][n - 2] * matrix[2][n - 3]
    matrix[2][n - 2] = matrix[2][n - 2] - matrix[0][n - 2] * matrix[3][n - 3]

    matrix[0][n - 1] = matrix[0][n - 1] / matrix[1][n - 2]
    matrix[1][n - 1] = matrix[1][n - 1] - matrix[0][n - 1] * matrix[2][n - 2]

    for i in range(1, n):
        x[i] = x[i] - matrix[0][i] * x[i - 1]

    x[n - 1] = x[n - 1] / matrix[1][n - 1]
    x[n - 2] = (x[n - 2] - matrix[2][n - 2] * x[n - 1]) / matrix[1][n - 2]

    for i in range(n - 3, -1, -1):
        x[i] = (x[i] - matrix[3][i] * x[i + 2] - matrix[2][i] * x[i + 1]) / matrix[1][i]

    determinant = 1
    for i in range(n):
        determinant *= matrix[1][i]

    return [x, determinant]


def createPlot(n_values_, fun, no_samples):
    execution_time = [0] * len(n_values_)

    for i in range(len(n_values_)):
        for sample in range(no_samples):
            star_time = time.time()
            fun(n_values_[i])
            end_time = time.time()
            fun_time = end_time - star_time
            execution_time[i] += fun_time
        execution_time[i] /= no_samples

    plt.plot(n_values_, execution_time, color="r")
    plt.xlabel('n')
    plt.ylabel('Czas (s)')
    plt.title('Zadanie 3')
    plt.grid(True)
    plt.show()

n = 124
print("Szukane rozwiązanie to: ", calculation(n)[0])
print()
print("Wyznacznik macierzy A = ", calculation(n)[1])

createPlot(list(range(124,1001)),calculation, 100)
