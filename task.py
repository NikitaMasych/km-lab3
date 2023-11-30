import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Задані початкові умови
y0 = [2]
t_span = [1, 4]
epsilon = 1e-4  # Задана точність

# Функція, що визначає диференціальне рівняння
def func(t, y):
    return t**2 + 2*y

t_eval = np.linspace(t_span[0], t_span[1], 100)  # Генеруємо 100 точок для побудови графіка

# Розв'язуємо рівняння за допомогою методу Рунге-Кутта-2
sol_rk2 = solve_ivp(func, t_span, y0, method='RK23', t_eval=t_eval)

# Розв'язуємо рівняння та отримуємо значення y за допомогою методу Мілна-Сімпсона
y_milne = np.zeros_like(t_eval)
y_milne[0] = y0[0]

milne_points = []  # Зберігаємо точки, де обчислюємо Мілн

for i in range(1, len(t_eval)):
    h = t_eval[i] - t_eval[i-1]
    k1 = h * func(t_eval[i-1], y_milne[i-1])
    k2 = h * func(t_eval[i-1] + h/2, y_milne[i-1] + k1/2)
    k3 = h * func(t_eval[i-1] + h/2, y_milne[i-1] + k2/2)
    k4 = h * func(t_eval[i], y_milne[i-1] + k3)
    
    y_milne[i] = y_milne[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    milne_points.append([t_eval[i-1], y_milne[i-1]])  # Зберігаємо точки для відображення

    # Перевірка точності і зупинка, якщо досягнута
    if np.abs(y_milne[i] - y_milne[i-1]) < epsilon:
        break


milne_points = np.array(milne_points)

# Побудова графіків
plt.figure(figsize=(10, 6))

#plt.plot(sol_rk2.t, sol_rk2.y[0], label='Runge-Kutta-2', color='black', linestyle='-')
plt.plot(t_eval, y_milne, label='Milne-Simpson', color='black', linestyle='-')

#plt.scatter(sol_rk2.t, sol_rk2.y[0], color='red', label='Runge-Kutta-2 points')
plt.scatter(milne_points[:, 0], milne_points[:, 1], color='orange', label='Milne-Simpson points')

plt.xlabel('t')
plt.ylabel('y')
plt.title('Розв\'язок диференціального рівняння')
plt.legend()
plt.grid(True)
plt.show()
