# Mateusz Kosowski 251558
# Jakub Rosiak 251620
import time
from math import pi

import matplotlib.pyplot as plt
import numpy as np

max_task_1 = 100
max_task_2 = 1.850547


def cooling_function(factor, temp):
    return factor * temp


def f1(x):
    if -105 < x < -95:
        return -2 * abs(x + 100) + 10
    if 95 < x < 105:
        return -2.2 * abs(x - 100) + 11
    return 0


def f2(x):
    return x * np.sin(10 * pi * x)


def generate_neighbor(solution, temperature, min_x, max_x):
    neighbor = solution + np.random.uniform(-2 * temperature, 2 * temperature)

    return np.clip(neighbor, min_x, max_x)


def simulated_annealing(func, initial_temp, min_x, max_x, cooling_factor, epochs, coefficient, steps):

    temperature = initial_temp
    best_solution = np.random.uniform(min_x, max_x)
    count_of_iterations = 0

    for _ in range(epochs):
        for _ in range(steps):
            neighbor = generate_neighbor(best_solution, temperature, min_x, max_x)

            delta = func(neighbor) - func(best_solution)

            if delta > 0:
                best_solution = neighbor

            else:
                x = np.random.uniform(0, 1)
                if x < np.exp(delta / (coefficient * temperature)):
                    best_solution = neighbor

            count_of_iterations += 1
        temperature = cooling_function(cooling_factor, temperature)
    return best_solution, count_of_iterations


def plot(f, min_x, max_x, amount, solution):
    x = np.linspace(min_x, max_x, amount)
    y = [f(xi) for xi in x]
    plt.plot(x, y)
    plt.scatter(solution, f(solution), color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


def generate_metrics(f, epochs, temperature, alpha, steps, min_x, max_x, coefficient, max_value):
    start = time.perf_counter()
    solution, iterations = simulated_annealing(func=f, initial_temp=temperature, min_x=min_x, max_x=max_x,
                                               cooling_factor=alpha, epochs=epochs, steps=steps,
                                               coefficient=coefficient)
    end = time.perf_counter()
    precision = abs(f(solution) - f(max_value))
    print(f"Rozwiązanie: {solution}")
    print(f"Czas wykonywania: {(end - start) * 1000:.2f} ms")
    print(f"Liczba iteracji: {iterations}")
    print(f"Dokładność: {precision}\n")
    return solution


def check_f1():
    epoch_iterations = [2000, 1000, 100]
    temperature = [400, 200, 600]
    alpha = [0.9, 0.94, 0.98]

    for Mi in epoch_iterations:
        for i in range(5):
            print(f"Wyniki dla ilości epok = {Mi} ({i + 1}/5)")
            generate_metrics(f=f1, epochs=Mi, temperature=500, alpha=0.999, steps=1, min_x=-150, max_x=150,
                             coefficient=0.1, max_value=max_task_1)

    for Ti in temperature:
        for i in range(5):
            print(f"Wyniki dla temperatury = {Ti} ({i + 1}/5)")
            generate_metrics(f=f1, epochs=3000, temperature=Ti, alpha=0.999, steps=1, min_x=-150, max_x=150,
                             coefficient=0.1, max_value=max_task_1)

    for ai in alpha:
        for i in range(5):
            print(f"Wyniki dla alpha = {ai} ({i + 1}/5)")
            generate_metrics(f=f1, epochs=3000, temperature=500, alpha=ai, steps=1, min_x=-150, max_x=150,
                             coefficient=0.1, max_value=max_task_1)


def check_f2():
    M = [1000, 800, 500]
    T = [10, 50, 75]
    a = [0.9, 0.94, 0.98]
    for Mi in M:
        for i in range(5):
            print(f"Wyniki dla M = {Mi} ({i + 1}/5)")
            generate_metrics(f=f2, epochs=Mi, temperature=5, alpha=0.997, steps=1, min_x=-1, max_x=2, coefficient=0.1,
                             max_value=max_task_2)

    for Ti in T:
        for i in range(5):
            print(f"Wyniki dla T = {Ti} ({i + 1}/5)")
            generate_metrics(f=f2, epochs=1200, temperature=Ti, alpha=0.997, steps=1, min_x=-1, max_x=2,
                             coefficient=0.1, max_value=max_task_2)

    for ai in a:
        for i in range(5):
            print(f"Wyniki dla a = {ai} ({i + 1}/5)")
            generate_metrics(f=f2, epochs=1200, temperature=5, alpha=ai, steps=1, min_x=-1, max_x=2, coefficient=0.1,
                             max_value=max_task_2)


if __name__ == '__main__':
    print("Rozwiązanie domyślne funkcji 1: ")
    s1 = generate_metrics(f=f1, epochs=3000, temperature=500, alpha=0.999, steps=1, min_x=-150, max_x=150,
                          coefficient=0.1, max_value=max_task_1)

    plot(f1, -150, 150, 500, s1)

    check_f1()
    print("Rozwiązanie domyślne funkcji 2: ")
    s2 = generate_metrics(f=f2, epochs=1200, temperature=5, alpha=0.997, steps=1, min_x=-1, max_x=2, coefficient=0.1,
                          max_value=max_task_2)
    plot(f2, -1, 2, 1000, s2)
    check_f2()
