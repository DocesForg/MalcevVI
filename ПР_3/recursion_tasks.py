# Файл: recursion_tasks.py

import os

# 1. Бинарный поиск с использованием рекурсии
def binary_search(arr, target, left, right):
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)

# Пример использования:
arr = [1, 3, 5, 7, 9, 11, 13]
target = 7
index = binary_search(arr, target, 0, len(arr) - 1)
print(f"Элемент {target} найден по индексу {index}")

# 2. Рекурсивный обход файловой системы
def list_files(directory, indent=0):
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        print(" " * indent + item)
        if os.path.isdir(path):
            list_files(path, indent + 4)

# Пример использования:
# Замените 'path_to_directory' на путь к каталогу
# list_files('path_to_directory')

# 3. Ханойские башни
def hanoi_towers(n, source, target, auxiliary):
    if n == 1:
        print(f"Перемещаем диск 1 со стержня {source} на стержень {target}")
        return
    hanoi_towers(n - 1, source, auxiliary, target)
    print(f"Перемещаем диск {n} со стержня {source} на стержень {target}")
    hanoi_towers(n - 1, auxiliary, target, source)

# Пример использования:
hanoi_towers(3, 'A', 'C', 'B')

# Добавляем функцию для измерения максимальной глубины рекурсии при обходе файловой системы
def list_files_with_depth(directory, indent=0, max_depth=0):
    current_depth = indent // 4
    max_depth = max(max_depth, current_depth)
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            max_depth = list_files_with_depth(path, indent + 4, max_depth)
    return max_depth

# Пример использования:
# max_depth = list_files_with_depth('path_to_directory')
# print(f"Максимальная глубина рекурсии: {max_depth}")