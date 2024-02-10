import matplotlib.pyplot as plt
import numpy as np

def power_iteration_method(matrix, tolerance=1e-6):
    n = len(matrix)
    x = [1.0] * n
    errors = []

    for i in range(100):
        y = x.copy()
        z = matrix_vector_multiplication(matrix, y)
        norm = sum(val ** 2 for val in z) ** 0.5
        x = vector_normalization(z)
        error = sum(abs(a - b) ** 2 for a, b in zip(y, x)) ** 0.5
        errors.append(error)

        if error < tolerance:
            break

    return norm, x, errors


matrix_A = np.array([
    [8, 1, 0, 0],
    [1, 7, 2, 0],
    [0, 2, 6, 3],
    [0, 0, 3, 5]
], dtype=float)

wartości_własne, wektory_własne = np.linalg.eig(matrix_A)

największa_wartość_własna = max(abs(wartości_własne))

print("Wartości własne macierzy M:", wartości_własne)
print("Największa co do modułu wartość własna:", największa_wartość_własna)

indeks_dominującej_wartości = np.argmax(abs(wartości_własne))

dominujący_wektor_własny = wektory_własne[:, indeks_dominującej_wartości]

print("Dominujący wektor własny:")
print(dominujący_wektor_własny)


def matrix_vector_multiplication(matrix, vector):
    return np.dot(matrix, vector)


def vector_normalization(vector):
    norm = sum(x ** 2 for x in vector) ** 0.5
    return [x / norm for x in vector]


eigenvalue, eigenvector, errors = power_iteration_method(matrix_A)
print("Dominujaca wartosc wlasna dla a: ", eigenvalue)
print("Wektor wlasny dla a: ", eigenvector)


def QR_algorithm_method(matrix):
    A = np.array(matrix)
    diagonal_errors = []

    for k in range(1, 120 + 1):
        Qk, Rk = np.linalg.qr(A)
        A = np.dot(Rk, Qk)
        eigenvalues = np.linalg.eigvals(A)
        error = np.abs(eigenvalues - np.diag(A))
        diagonal_errors.append(error)

        if np.all(np.abs(np.diag(A)) < 1e-10):
            break

    return diagonal_errors, eigenvalues


diagonal_errors, eigenvalues = QR_algorithm_method(matrix_A)
diagonal_errors = np.array(diagonal_errors)

for i in range(len(matrix_A)):
    plt.plot(range(1, len(diagonal_errors) + 1), diagonal_errors[:, i], label = "Element:"+str(i)+","+str(i))

plt.yscale('log')
plt.xlabel('Iteracje')
plt.ylabel('Roznica')
plt.title('Roznica pomiedzy wartosciami wlasnymi,\n a elementrami przekatnymi dla macierzy A')
plt.legend()
plt.show()

print("Wartości wlasne dla b: ", eigenvalues)


def wilkinson_algorithm(matrix, tolerance=1e-6):
    n = len(matrix)
    x = [1.0] * n
    eigenvalues = []

    for i in range(150):
        y = x.copy()
        z = matrix_vector_multiplication(matrix, y)
        norm = sum(val ** 2 for val in z) ** 0.5
        x = vector_normalization(z)
        another_eigenvalue = sum(x[i] * sum(matrix[i][j] * x[j] for j in range(n)) for i in range(n)) / sum(
            x[i] ** 2 for i in range(n))
        eigenvalues.append(0.5 * (another_eigenvalue + eigenvalue))

        if norm < tolerance:
            break

    return eigenvalues


eigenvalue, eigenvector, errors = power_iteration_method(matrix_A)
plt.plot(range(1, len(errors) + 1), errors, label='Metoda Potegowa')

p_values = wilkinson_algorithm(matrix_A)
for i in range(len(matrix_A)):
    matrix_A[i][i] = abs(matrix_A[i][i] - p_values[i])
eigenvalue_1, eigenvector_1, errors_1 = power_iteration_method(matrix_A)
plt.plot(range(1, len(errors_1) + 1), errors_1, label='Metoda Wilkinsona', c="red")

plt.grid(True)
plt.yscale('log')
plt.xlabel('Iteracje')
plt.ylabel('Blad zbieznosci')
plt.title('Zbieznosc metody potegowej i Wilkinsona')
plt.legend()
plt.show()
