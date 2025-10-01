# Файл: memoization.py

import time
import matplotlib.pyplot as plt

# Мемоизированная версия для вычисления чисел Фибоначчи
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]

# Сравнение производительности наивной и мемоизированной версий для n=35
from recursion import fibonacci  # Импортируем функцию из recursion.py

n = 35

# Наивная версия
start_time = time.time()
result_naive = fibonacci(n)  # Из файла recursion.py
end_time = time.time()
print(f"Наивная рекурсия (n={n}): {end_time - start_time:.6f} сек")

# Мемоизированная версия
start_time = time.time()
result_memo = fibonacci_memo(n)
end_time = time.time()
print(f"Мемоизация (n={n}): {end_time - start_time:.6f} сек")

# Замеры времени выполнения для чисел Фибоначчи
times_naive = []
times_memo = []

for n in range(1, 36):
    start_time = time.time()
    fibonacci(n)  # Наивная рекурсия
    times_naive.append(time.time() - start_time)

    start_time = time.time()
    fibonacci_memo(n)  # Мемоизация
    times_memo.append(time.time() - start_time)

# Построение графика
plt.plot(range(1, 36), times_naive, label="Наивная рекурсия")
plt.plot(range(1, 36), times_memo, label="Мемоизация")
plt.xlabel("n")
plt.ylabel("Время выполнения (сек)")
plt.title("Сравнение времени выполнения")
plt.legend()

plt.savefig("fibonacci_comparison.png") 

plt.show()