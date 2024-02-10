import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = []
y = []

with open('dane', 'r') as file:
    lines = file.readlines()
    for line in lines:
        values = line.strip().split(',')
        x.append(float(values[0]))
        y.append(float(values[1]))

def F(x, a, b, c, d):
    return a * x ** 2 + b * np.sin(x) + c * np.cos(5 * x) + d * np.exp(-x)

def G(x, a, b, c, d):
    return a * x**2 + b * np.sin(-3*x) + c * np.cos(2*x) + d * np.exp(-7*x)

def coefficients(X, y):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X_inv = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T
    return X_inv @ y

poprawne = curve_fit(F, x, y)
print("Poprawne wartosci: " + str(poprawne[0]))

x_data = np.array(x)

X = np.column_stack([x_data ** 2, np.sin(x_data), np.cos(5 * x_data), np.exp(-x_data)])
coefficients_values = coefficients(X, y)

a, b, c, d = coefficients_values

print(f'Optymalne wartości dla F(x): a: {a}, b: {b}, c: {c}, d: {d}')

fx_plot = np.linspace(min(x_data), max(x_data), 1000)
fy_plot = F(fx_plot, a, b, c, d)

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y, color="red", marker="o", label='Punkty')
plt.plot(fx_plot, fy_plot, label='Dopasowana F(x)')
plt.xlabel('x')
plt.ylabel('Y')
plt.title('Wykres dla F(x)')
plt.legend()
plt.grid(True)
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, (num_points, noise_std, noise_mean) in enumerate([(20, 3, 1), (100, 3, 1), (300, 3, 1)]):
    x_data = np.linspace(0, 10, num_points)
    delta_y = np.random.normal(noise_mean, noise_std, len(x_data))
    y_val = G(x_data, 0.1, 0.2, 0.3, 0.4)
    y_noisy = y_val + delta_y
    X = np.column_stack([x_data ** 2, np.sin(-3 * x_data), np.cos(2 * x_data), np.exp(-7 * x_data)])
    coefficients_values = coefficients(X, y_noisy)
    a, b, c, d = coefficients_values
    print(f'Optymalne wartości dla G(x): a: {a}, b: {b}, c: {c}, d: {d}')

    x_plot = np.linspace(min(x_data), max(x_data), 1000)
    y_plot = G(x_plot, *coefficients_values)

    axs[i].plot(x_plot, y_plot, label='Dopasowana G(x)', color='blue')
    axs[i].plot(x_plot, G(x_plot, 0.1, 0.2, 0.3, 0.4), label='Oryginalna G(x)', color='green', linestyle='--')
    axs[i].scatter(x_data, y_noisy, color="red", marker="o", label='Punkty')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('Y')
    axs[i].set_title(
        f'Wykres dla G(x) z {num_points} punktami\n Średnia szumu={noise_mean}\n Odchylenie standardowe={noise_std}')
    axs[i].legend()
    axs[i].grid(True)

print("Zadane wartości dla G(x): a: 0.1 b: 0.2 c: 0.3 d: 0.4")
plt.tight_layout()

plt.show()