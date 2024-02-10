import math
import matplotlib.pyplot as plt

def f(x):
    return math.sin(x) - 0.4

def g(x):
    return (math.sin(x) - 0.4) ** 2

def u(x):
    return (math.sin(x) - 0.4) ** 2 / (2 * (math.sin(x) - 0.4) * math.cos(x))

def du(x):
    return (5 - 2 * math.sin(x)) / (10 * (math.cos(x) ** 2))

def bisection_method(f, a, b, tol):
    num_steps = 0
    x_values = []
    x_star = math.asin(0.4)

    while (b - a) / 2.0 > tol:
        midpoint = (a + b) / 2.0
        x_values.append(abs(midpoint - x_star))
        if f(midpoint) == 0:
            return midpoint, num_steps, x_values
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        num_steps += 1

    return (a + b) / 2.0, num_steps, x_values

def regula_falsi_method(f, a, b, tol, max_iter=1000):
    num_steps = 0
    x_values = []
    x_star = math.asin(0.4)

    for i in range(max_iter):
        f_a = f(a)
        f_b = f(b)
        if abs(f_a) < tol:
            return a, num_steps, x_values
        if abs(f_b) < tol:
            return b, num_steps, x_values
        midpoint = (a * f_b - b * f_a) / (f_b - f_a)
        x_values.append(abs(midpoint - x_star))
        f_mid = f(midpoint)
        if abs(f_mid) < tol:
            return midpoint, num_steps, x_values
        if f_a * f_mid < 0:
            b = midpoint
        else:
            a = midpoint
        if abs(b - a) < tol:
            return midpoint, num_steps, x_values
        num_steps += 1

def secant_method(f, x0, x1, tol):
    num_steps = 0
    x_values = []
    x_star = math.asin(0.4)

    while True:
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x_values.append(abs(x2 - x_star))
        if f(x2) == 0 or abs(f(x2)) < tol:
            return x2, num_steps, x_values
        x0, x1 = x1, x2
        num_steps += 1

def newton_method(f, df, x0, tol):
    num_steps = 0
    x_values = []
    x_star = math.asin(0.4)

    while True:
        x1 = x0 - f(x0) / df(x0)
        x_values.append(abs(x1 - x_star))
        if f(x1) == 0 or abs(f(x1)) < tol:
            return x1, num_steps, x_values
        x0 = x1
        num_steps += 1

a = 0
b = math.pi / 2
tolerance = 1e-16
tolerance_v2 = 1e-10

# Wykres dla funkcji f(x)
result_bisection_f, steps_bisection_f, x_values_bisection_f = bisection_method(f, a, b, tolerance)
result_regula_falsi_f, steps_regula_falsi_f, x_values_regula_falsi_f = regula_falsi_method(f, a, b, tolerance)
result_secant_f, steps_secant_f, x_values_secant_f = secant_method(f, a, b, tolerance)
result_newton_f, steps_newton_f, x_values_newton_f = newton_method(f, lambda x: math.cos(x), 0, tolerance)

plt.figure(figsize=(10, 6))
plt.plot(range(len(x_values_bisection_f)), x_values_bisection_f, label='Bisekcja (f(x))')
plt.plot(range(len(x_values_regula_falsi_f)), x_values_regula_falsi_f, label='Regula Falsi (f(x))')
plt.plot(range(len(x_values_secant_f)), x_values_secant_f, label='Sieczne (f(x))')
plt.plot(range(len(x_values_newton_f)), x_values_newton_f, label='Newton (f(x))')
plt.yscale('log')
plt.xlabel('Liczba kroków')
plt.ylabel('Wartość |xi - x∗|')
plt.legend()
plt.title('Zmiana |xi - x∗| dla funkcji f(x)')
plt.show()

# Wykres dla funkcji g(x)
result_secant_g, steps_secant_g, x_values_secant_g = secant_method(g, a, b/2, tolerance)
result_newton_g, steps_newton_g, x_values_newton_g = newton_method(g, lambda x: 2 * (math.sin(x) - 0.4) * math.cos(x),
                                                                   0, tolerance)

plt.figure(figsize=(10, 6))
plt.plot(range(len(x_values_secant_g)), x_values_secant_g, label='Sieczne (g(x))')
plt.plot(range(len(x_values_newton_g)), x_values_newton_g, label='Newton (g(x))')
plt.yscale('log')
plt.xlabel('Liczba kroków')
plt.ylabel('Wartość |xi - x∗|')
plt.legend()
plt.title('Zmiana |xi - x∗| dla funkcji g(x)')
plt.show()

# Wykres dla funkcji u(x)
result_secant_u, steps_secant_u, x_values_secant_u = secant_method(u, a, b, tolerance_v2)
result_newton_u, steps_newton_u, x_values_newton_u = newton_method(u, du, 0, tolerance_v2)
result_bisection_u, steps_bisection_u, x_values_bisection_u = bisection_method(u, a, b, tolerance_v2)

plt.figure(figsize=(10, 6))
plt.plot(range(len(x_values_secant_u)), x_values_secant_u, label='Sieczne (u(x))')
plt.plot(range(len(x_values_newton_u)), x_values_newton_u, label='Newton (u(x))')
plt.plot(range(len(x_values_bisection_u)), x_values_bisection_u, label='Bisekcja (u(x))')
plt.yscale('log')
plt.xlabel('Liczba kroków')
plt.ylabel('Wartość |xi - x∗|')
plt.legend()
plt.title('Zmiana |xi - x∗| dla funkcji u(x)')
plt.show()

print("Wyniki dla funkcji f(x):")
print("Bisekcja:", result_bisection_f, "Liczba kroków:", steps_bisection_f)
print("Regula Falsi:", result_regula_falsi_f, "Liczba kroków:", steps_regula_falsi_f)
print("Sieczne:", result_secant_f, "Liczba kroków:", steps_secant_f)
print("Newton:", result_newton_f, "Liczba kroków:", steps_newton_f)

print("\nWyniki dla funkcji g(x):")
print("Sieczne:", result_secant_g, "Liczba kroków:", steps_secant_g)
print("Newton:", result_newton_g, "Liczba kroków:", steps_newton_g)

print("\nWyniki dla funkcji u(x):")
print("Sieczne:", result_secant_u, "Liczba kroków:", steps_secant_u)
print("Newton:", result_newton_u, "Liczba kroków:", steps_newton_u)
print("Bisekcja:", result_bisection_u, "Liczba kroków:", steps_bisection_u)
