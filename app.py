# Mateusz Kosowski 251558
# Jakub Rosiak 251620

import time
from math import pi
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime

max_task_1 = 100
max_task_2 = 1.850547


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

    history = []
    temp_history = []

    for epoch in range(epochs):
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
            history.append((best_solution, func(best_solution)))
            temp_history.append(temperature)

        temperature = temperature * cooling_factor

    return best_solution, count_of_iterations, history, temp_history


def plot_function_with_solution(f, min_x, max_x, amount, solution, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(min_x, max_x, amount)
    y = [f(xi) for xi in x]
    ax.plot(x, y, label='Funkcja', linewidth=2)
    ax.scatter(solution, f(solution), color='red', s=100, zorder=5, label=f'Rozwiązanie: x={solution:.4f}')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_convergence(history, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = list(range(len(history)))
    values = [h[1] for h in history]
    ax.plot(iterations, values, linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Iteracja', fontsize=12)
    ax.set_ylabel('Wartość funkcji celu', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return fig


def plot_temperature(temp_history, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = list(range(len(temp_history)))
    ax.plot(iterations, temp_history, linewidth=1.5, color='orange')
    ax.set_xlabel('Iteracja', fontsize=12)
    ax.set_ylabel('Temperatura', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return fig


def save_figure_to_bytes(fig, format='png', dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf


def main():
    st.set_page_config(
        page_title="Symulowane Wyżarzanie",
        layout="wide"
    )

    st.title("Algorytm Symulowanego Wyżarzania")
    st.markdown("**Autorzy:** Mateusz Kosowski 251558, Jakub Rosiak 251620")

    # Sidebar - wybór funkcji
    st.sidebar.header("Konfiguracja")

    function_choice = st.sidebar.selectbox(
        "Wybierz funkcję do optymalizacji",
        ["Funkcja 1 (dwuwierzchołkowa)", "Funkcja 2 (sinusoidalna)"]
    )

    # Ustawienia domyślne w zależności od wybranej funkcji
    if function_choice == "Funkcja 1 (dwuwierzchołkowa)":
        func = f1
        default_epochs = 3000
        default_temp = 500.0
        default_alpha = 0.999
        default_min_x = -150.0
        default_max_x = 150.0
        max_value = max_task_1
        plot_points = 500
    else:
        func = f2
        default_epochs = 1200
        default_temp = 5.0
        default_alpha = 0.997
        default_min_x = -1.0
        default_max_x = 2.0
        max_value = max_task_2
        plot_points = 1000

    st.sidebar.subheader("Parametry algorytmu")

    col1, col2 = st.sidebar.columns(2)

    with col1:
        epochs = st.number_input("Liczba epok", min_value=1, max_value=10000, value=default_epochs, step=100)
        min_x = st.number_input("Min X", value=default_min_x, step=0.1)

    with col2:
        steps = st.number_input("Kroki na epokę", min_value=1, max_value=100, value=1, step=1)
        max_x = st.number_input("Max X", value=default_max_x, step=0.1)

    temperature = st.sidebar.slider(
        "Temperatura początkowa",
        min_value=1.0,
        max_value=1000.0,
        value=default_temp,
        step=1.0
    )

    alpha = st.sidebar.slider(
        "Współczynnik chłodzenia (α)",
        min_value=0.8,
        max_value=0.999,
        value=default_alpha,
        step=0.001,
        format="%.3f"
    )

    coefficient = st.sidebar.slider(
        "Współczynnik akceptacji",
        min_value=0.01,
        max_value=0.1,
        value=0.01,
        step=0.01
    )

    # Przycisk uruchomienia
    run_button = st.sidebar.button("Uruchom algorytm", type="primary")

    # Sekcja informacyjna
    with st.expander("Informacje o funkcjach"):
        st.markdown("""
        ### Funkcja 1 (dwuwierzchołkowa)
        Funkcja z dwoma lokalnymi maksimami:
        - Maksimum globalne w x ≈ 100 z wartością ≈ 11
        - Maksimum lokalne w x ≈ -100 z wartością ≈ 10
        
        ### Funkcja 2 (sinusoidalna)
        Funkcja: f(x) = x * sin(10πx) na przedziale [-1, 2]
        - Wielokrotne lokalne maksima i minima
        - Maksimum globalne ≈ 1.850547
        """)

    with st.expander("Informacje o parametrach"):
        st.markdown("""
        - **Liczba epok**: Liczba głównych iteracji algorytmu
        - **Kroki na epokę**: Liczba prób w każdej epoce przed obniżeniem temperatury
        - **Temperatura początkowa**: Wyższa temperatura = większa eksploracja na początku
        - **Współczynnik chłodzenia (α)**: Jak szybko temperatura spada (bliżej 1 = wolniejsze chłodzenie)
        - **Współczynnik akceptacji**: Wpływa na prawdopodobieństwo akceptacji gorszych rozwiązań
        """)

    # Główna logika
    if run_button:
        with st.spinner('Wykonywanie algorytmu symulowanego wyżarzania...'):
            start = time.perf_counter()

            solution, iterations, history, temp_history = simulated_annealing(
                func=func,
                initial_temp=temperature,
                min_x=min_x,
                max_x=max_x,
                cooling_factor=alpha,
                epochs=epochs,
                steps=steps,
                coefficient=coefficient
            )

            end = time.perf_counter()
            execution_time = (end - start) * 1000
            precision = abs(func(solution) - func(max_value))

        # Wyświetlanie wyników
        st.success("Algorytm zakończony!")

        # Metryki
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rozwiązanie (x)", f"{solution:.6f}")

        with col2:
            st.metric("Wartość f(x)", f"{func(solution):.6f}")

        with col3:
            st.metric("Czas wykonania", f"{execution_time:.2f} ms")

        with col4:
            st.metric("Liczba iteracji", iterations)

        st.metric("Dokładność (odległość od optimum)", f"{precision:.6f}")

        # Wykresy
        st.subheader("Wizualizacje")

        tab1, tab2, tab3 = st.tabs(["Funkcja z rozwiązaniem", "Zbieżność", "Temperatura"])

        with tab1:
            fig1 = plot_function_with_solution(
                func, min_x, max_x, plot_points, solution,
                f"Funkcja z zaznaczonym rozwiązaniem"
            )
            st.pyplot(fig1)

            # Przyciski do pobierania
            col_btn1 = st.columns(1)[0]
            with col_btn1:
                buf1_png = save_figure_to_bytes(fig1, format='png', dpi=300)
                st.download_button(
                    label="Pobierz",
                    data=buf1_png,
                    file_name=f"funkcja_rozwiazanie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

            plt.close(fig1)

        with tab2:
            fig2 = plot_convergence(history, "Zbieżność algorytmu")
            st.pyplot(fig2)

            st.info(f"Najlepsza wartość początkowa: {history[0][1]:.6f}")
            st.info(f"Najlepsza wartość końcowa: {history[-1][1]:.6f}")

            # Przyciski do pobierania
            col_btn1 = st.columns(1)[0]
            with col_btn1:
                buf2_png = save_figure_to_bytes(fig2, format='png', dpi=300)
                st.download_button(
                    label="Pobierz",
                    data=buf2_png,
                    file_name=f"zbieznosc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

            plt.close(fig2)

        with tab3:
            fig3 = plot_temperature(temp_history, "Spadek temperatury w czasie")
            st.pyplot(fig3)

            st.info(f"Temperatura początkowa: {temp_history[0]:.2f}")
            st.info(f"Temperatura końcowa: {temp_history[-1]:.6f}")

            # Przyciski do pobierania
            col_btn1 = st.columns(1)[0]
            with col_btn1:
                buf3_png = save_figure_to_bytes(fig3, format='png', dpi=300)
                st.download_button(
                    label="Pobierz",
                    data=buf3_png,
                    file_name=f"temperatura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

            plt.close(fig3)

        # Szczegóły techniczne
        with st.expander("Szczegóły techniczne"):
            st.markdown(f"""
            **Parametry użyte:**
            - Funkcja: {function_choice}
            - Zakres: [{min_x}, {max_x}]
            - Liczba epok: {epochs}
            - Kroki na epokę: {steps}
            - Temperatura początkowa: {temperature}
            - Współczynnik chłodzenia: {alpha}
            - Współczynnik akceptacji: {coefficient}
            
            **Wyniki:**
            - Rozwiązanie: x = {solution:.6f}
            - Wartość funkcji: f(x) = {func(solution):.6f}
            - Wartość optimum: {max_value:.6f}
            - Precyzja: {precision:.6f}
            - Całkowita liczba iteracji: {iterations}
            - Czas wykonania: {execution_time:.2f} ms
            """)

    else:
        st.info("Skonfiguruj parametry w panelu bocznym i kliknij 'Uruchom algorytm'")

        st.subheader("Podgląd wybranej funkcji")
        fig_preview = plot_function_with_solution(
            func, default_min_x, default_max_x, plot_points,
            (default_min_x + default_max_x) / 2,
            f"Podgląd: {function_choice}"
        )
        st.pyplot(fig_preview)
        plt.close(fig_preview)


if __name__ == '__main__':
    main()

