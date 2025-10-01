# Файл: recursion.py

import sys

# 1. Вычисление факториала числа n
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

# Временная сложность: O(n)
# Глубина рекурсии: n

# 2. Вычисление n-го числа Фибоначчи
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Временная сложность: O(2^n) (экспоненциальная)
# Глубина рекурсии: n

# 3. Быстрое возведение числа a в степень n
def fast_power(a, n):
    if n == 0:
        return 1
    elif n % 2 == 0:
        half = fast_power(a, n // 2)
        return half * half
    else:
        return a * fast_power(a, n - 1)

# Временная сложность: O(log n)
# Глубина рекурсии: log n