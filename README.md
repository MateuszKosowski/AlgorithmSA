# Algorytm Symulowanego Wyżarzania - Interaktywna Aplikacja Streamlit

**Autorzy:** Mateusz Kosowski 251558, Jakub Rosiak 251620

## Opis

Interaktywna aplikacja Streamlit do wizualizacji i testowania algorytmu symulowanego wyżarzania na dwóch funkcjach testowych.

## Funkcje

### Funkcja 1 (dwuwierzchołkowa)
- Dwuwierzchołkowa funkcja z dwoma maksimami lokalnymi
- Maksimum globalne w x ≈ 100 z wartością ≈ 11
- Maksimum lokalne w x ≈ -100 z wartością ≈ 10

### Funkcja 2 (sinusoidalna)
- f(x) = x * sin(10πx) na przedziale [-1, 2]
- Wielokrotne lokalne maksima i minima
- Maksimum globalne ≈ 1.850547

## Instalacja

1. Upewnij się, że masz zainstalowanego Pythona (wersja 3.8 lub nowsza)

2. Zainstaluj wymagane biblioteki:
```bash
pip install -r requirements.txt
```

## Uruchomienie

### Aplikacja Streamlit (interaktywna)
```bash
streamlit run app.py
```

### Wersja konsolowa (oryginalna)
```bash
python main.py
```

## Funkcjonalności aplikacji Streamlit

- **Wybór funkcji** - możliwość wyboru między dwiema funkcjami testowymi
- **Konfiguracja parametrów** - interaktywne suwaki do ustawiania:
  - Liczby epok
  - Temperatury początkowej
  - Współczynnika chłodzenia (α)
  - Współczynnika akceptacji
  - Zakresu przeszukiwania
- **Wizualizacje**:
  - Wykres funkcji z zaznaczonym znalezionym rozwiązaniem
  - Wykres zbieżności algorytmu
  - Wykres spadku temperatury w czasie
- **Metryki**:
  - Znalezione rozwiązanie
  - Wartość funkcji celu
  - Czas wykonania
  - Liczba iteracji
  - Dokładność (odległość od optimum)

## Parametry algorytmu

- **Liczba epok** - liczba głównych iteracji algorytmu
- **Kroki na epokę** - liczba prób w każdej epoce przed obniżeniem temperatury
- **Temperatura początkowa** - wyższa temperatura = większa eksploracja na początku
- **Współczynnik chłodzenia (α)** - jak szybko temperatura spada (bliżej 1 = wolniejsze chłodzenie)
- **Współczynnik akceptacji** - wpływa na prawdopodobieństwo akceptacji gorszych rozwiązań

## Struktura projektu

- `app.py` - aplikacja Streamlit z interfejsem graficznym
- `main.py` - oryginalna wersja konsolowa z testami
- `requirements.txt` - lista wymaganych bibliotek
- `README.md` - dokumentacja projektu

## Zachowana logika z oryginalnego kodu

Aplikacja Streamlit zachowuje wszystkie kluczowe implementacje:
- Funkcja `simulated_annealing()` - główny algorytm
- Funkcja `cooling_function()` - funkcja chłodzenia
- Funkcja `generate_neighbor()` - generowanie sąsiednich rozwiązań
- Funkcje `f1()` i `f2()` - funkcje testowe
- Wszystkie formuły matematyczne i logika algorytmu

## Licencja

Projekt stworzony na potrzeby kursu Metaheurystyki.

