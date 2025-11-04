# Mateusz Kosowski 251558
# Jakub Rosiak 251620
import math
import numpy as np
import matplotlib.pyplot as plt

def cooling_function(factor, temp):
    return factor * temp

def f1(x):
    if x > -105 and x < -95:
        return -2 * abs(x+100) + 10
    if x > 95 and x < 105:
        return -2.2 * abs(x - 100) + 11
    return 0

def f2(x):
    return x * math.sin(10 * math.pi * x)

def generate_neighbor(solution, coefficient):
    return solution + np.random.uniform(-coefficient, coefficient)

def simulated_annealing(func, initial_temp, initial_solution, cooling_factor, iterations, coefficient, min_temp=1e-3):

    # Energia początkowa na chodzenie po górach
    temperature = initial_temp

    # Losowy punkt startowy
    current_solution = initial_solution

    # Najlepsze rozwiązanie
    best_solution = initial_solution

    # Dopóki nasz poziom energii jest większy niż minimalny
    while temperature > min_temp:

        # Generowanie pobliskich możliwych kroków
        for _ in range(iterations):

            # Losowy punkt w zasięgu kroku
            neighbor = generate_neighbor(current_solution, coefficient)

            # Różnica funkcji celu (gdzie jest wyżej)
            delta = func(current_solution) - func(neighbor)

            # Jeśli jest lepiej, idziemy tam
            if delta < 0:
                current_solution = neighbor

            # Jeśli jest gorzej, idziemy tam z pewnym prawdopodobieństwem, że może być lepiej później
            else:

                # Rzut kostką (element losowy)
                x = np.random.uniform(0, 1)

                # Magiczny wzór na prawdopodobieństwo
                # Im mamy więcej sił tym chętniej zaryzykujemy
                # Im mniejsza różnica wysokości tym chętniej zaryzykujemy
                if x < np.exp(-delta / (coefficient * temperature)):
                    current_solution = neighbor

            # Aktualizacja najwyższego znalezionego punktu
            if func(current_solution) > func(best_solution):
                best_solution = current_solution

        # Raz na LiczbaIteracjiKorków tracimy trochę energii
        temperature = cooling_function(cooling_factor, temperature)
        print(temperature)
    return best_solution

def plot(f, min_x, max_x, amount, solution):
    x = np.linspace(min_x, max_x, amount)
    y = [f(xi) for xi in x]

    plt.plot(x, y)
    plt.scatter(solution, f(solution), color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

if __name__ == '__main__':
    solution1 = simulated_annealing(f1, 500, np.random.uniform(-150, 150), 0.999, 3000, 0.1, 1)
    plot(f1, -150, 150, 500, solution1)
    solution2 = simulated_annealing(f2, 5, np.random.uniform(-1, 1), 0.997, 1200, 0.1, 1)
    plot(f2, -10, 10, 10000, solution2)




