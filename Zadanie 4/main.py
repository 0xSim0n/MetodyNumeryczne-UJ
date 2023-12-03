import matplotlib.pyplot as plt
import time
import numpy as np


def sherman_morrison(n):
    b_vector = [5] * n
    matrix_M = [[11] * n, [7] * (n - 1) + [0]]

    z_vector = [0] * n
    x_vector = [0] * n

    z_vector[n - 1] = b_vector[n - 1] / matrix_M[0][n - 1]
    x_vector[n - 1] = 1 / matrix_M[0][n - 1]

    for i in range(n - 2, -1, -1):
        z_vector[i] = (b_vector[n - 2] - matrix_M[1][i] * z_vector[i + 1]) / matrix_M[0][i]
        x_vector[i] = (1 - matrix_M[1][i] * x_vector[i + 1]) / matrix_M[0][i]

    delta = sum(z_vector) / (1 + sum(x_vector))

    result_vector = [z_vector[i] - x_vector[i] * delta for i in range(len(z_vector))]
    return result_vector


def numpy_solve(n):
    A = np.ones((n, n))
    A += np.diag([11] * n)
    A += np.diag([7] * (n - 1), 1)
    b_vector = [5] * n
    x = np.linalg.solve(A, b_vector)
    print(x)


def calculate_execution_time(func, n, probes):
    execution_times = []

    for _ in range(probes):
        start_time = time.time()
        func(n)
        end_time = time.time()
        execution_times.append(end_time - start_time)

    average_time = sum(execution_times) / probes
    return average_time


def calculate_execution_time_sherman_morrison(n, probes):
    return calculate_execution_time(sherman_morrison, n, probes)


def calculate_execution_time_numpy(n, probes):
    return calculate_execution_time(numpy_solve, n, probes)


def plot_complexity(N_values, execution_times_sherman, execution_times, title):
    plt.plot(N_values, execution_times_sherman, label="Sherman-Morrison")
    plt.plot(N_values, execution_times, label="Numpy")
    plt.xlabel('n')
    plt.yscale('log')
    plt.ylabel('Time')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_complexity_v2(N_values, execution_times, title):
    plt.plot(N_values, execution_times, marker='*')
    plt.xlabel('n')
    plt.ylabel('Time')
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("Custom Implementation:")
    result_n_80 = sherman_morrison(80)
    print(result_n_80)

    print("Using Numpy Library:")
    numpy_solve(80)

    N_values_sherman_morrison = list(range(80, 3000, 100))
    execution_times_sherman_morrison = [calculate_execution_time_sherman_morrison(n, probes=50) for n in N_values_sherman_morrison]
    execution_times = [calculate_execution_time_numpy(n, probes=2) for n in N_values_sherman_morrison]
    plot_complexity(N_values_sherman_morrison, execution_times_sherman_morrison, execution_times, 'Sherman-Morrison vs Numpy')
    plot_complexity_v2(N_values_sherman_morrison, execution_times_sherman_morrison, 'Sherman-Morrison')
    plot_complexity_v2(N_values_sherman_morrison, execution_times, 'Numpy')
