import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.float32(np.sin(x**2))

def f_derivative(x):
    return np.float32(np.cos(x**2) * 2 * x)

def f_difference_a(f, x, h):
    return np.float32((f(x + h) - f(x)) / h)

def f_difference_b(f, x, h):
    return np.float32((f(x + h) - f(x - h)) / (2 * h))

def g(x):
    return np.sin(x**2)

def f_derivative_double(x):
    return np.cos(x**2) * 2 * x

def f_difference_a_double(f, x, h):
    return (f(x + h) - f(x)) / h

def f_difference_b_double(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

z = 200 # ilość zmian parametru h
x = 0.2

h_range_float = np.logspace(-7, 0, z)
a_float_errors = np.zeros(z)
b_float_errors = np.zeros(z)

h_range_double = np.logspace(-16, 0, z)
a_double_errors = np.zeros(z)
b_double_errors = np.zeros(z)

for i,h in enumerate(h_range_double):
    a_double_difference = f_difference_a_double(g, x, np.float64(h))
    b_double_difference = f_difference_b_double(g, x, np.float64(h))
    a_double_errors[i] = np.abs(a_double_difference - f_derivative_double(x))
    b_double_errors[i] = np.abs(b_double_difference - f_derivative_double(x))

for i,h in enumerate(h_range_float):
    a_float_difference = f_difference_a(f, x, np.float32(h))
    b_float_difference = f_difference_b(f, x, np.float32(h))
    a_float_errors[i] = np.abs(a_float_difference - f_derivative(x))
    b_float_errors[i] = np.abs(b_float_difference - f_derivative(x))


#dla a i b float
plt.figure(figsize=(6, 5))
plt.loglog(h_range_float, a_float_errors, label='float(a) error')
plt.loglog(h_range_float, b_float_errors, label='float(b) error')

plt.xlabel('h')
plt.ylabel('|Dhf(x) - f\'(x)|')
plt.legend()
plt.grid()
plt.title('Error for a and b type float')

#dla a i b double
plt.figure(figsize=(6, 5))
plt.loglog(h_range_double, a_double_errors, label='double(a) error')
plt.loglog(h_range_double, b_double_errors, label='double(b) error')

plt.xlabel('h')
plt.ylabel('|Dhf(x) - f\'(x)|')
plt.legend()
plt.grid()
plt.title('Error for a and b type double')
plt.show()


