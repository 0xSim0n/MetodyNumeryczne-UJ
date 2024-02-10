import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix


def jacobi(A, b, x0=None, epsilon=1e-10, max_iterations=1000):
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.linalg.inv(D)
    x = x0 if x0 is not None else np.zeros_like(b)
    history = [x]

    for _ in range(max_iterations):
        x_new = D_inv @ (b - R @ x)
        history.append(x_new)
        if np.linalg.norm(x_new - x) < epsilon:
            break
        x = x_new

    return np.array(history)


def gauss_seidel(A, b, x0=None, epsilon=1e-10, max_iterations=1000):
    x = x0 if x0 is not None else np.zeros_like(b)
    history = [x]

    for _ in range(max_iterations):
        x_new = np.copy(x)
        for i in range(len(A)):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

        history.append(x_new)
        if np.linalg.norm(x_new - x) < epsilon:
            break
        x = x_new

    return np.array(history)


N = 124
value = 2500.0
A = np.diag([3] * N) + np.diag([1] * (N - 1), k=1) + np.diag([1] * (N - 1), -1) + np.diag([0.15] * (N - 2),
                                                                                          k=2) + np.diag(
    [0.15] * (N - 2), -2)

b = np.array([2] + [3] * (N - 2) + [N])
x0 = np.full(N, value)
x_exact = np.linalg.solve(A, b)

history_jacobi = jacobi(A, b, x0)
history_gauss_seidel = gauss_seidel(A, b, x0)

iterations = range(len(history_jacobi))
errors_jacobi = np.linalg.norm(history_jacobi - x_exact, axis=1)
iterations_gs = range(len(history_gauss_seidel))
errors_gauss_seidel = np.linalg.norm(history_gauss_seidel - x_exact, axis=1)

plt.figure(figsize=(14, 7))
plt.plot(iterations, errors_jacobi, label='Metoda Jacobiego')
plt.plot(iterations_gs, errors_gauss_seidel, label='Metoda Gaussa-Seidela')
plt.yscale('log')
plt.xlabel('Iteracja')
plt.ylabel('Błąd względem rozwiązania dokładnego')
plt.title('Porównanie zbieżności metody Jacobiego i Gaussa-Seidela')
plt.legend()
plt.grid(True)
plt.show()

A_sparse_csc = csc_matrix(A)
x_jacobi = history_jacobi[-1]
x_gauss_seidel = history_gauss_seidel[-1]

x_exact_sparse = spsolve(A_sparse_csc, b)
print("Dokładne rozwiązanie:", x_exact_sparse[:5])
print("Rozwiązanie metodą Jacobiego:", x_jacobi[:5])
print("Rozwiązanie metodą Gaussa-Seidela:", x_gauss_seidel[:5])


