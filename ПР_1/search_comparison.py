

# Импорт необходимых библиотек
import matplotlib.pyplot as plt
import timeit


def linear_search(arr, target):
    """
    Линейный поиск элемента в массиве.
    Возвращает индекс target или -1, если не найден.
    Сложность: O(n), где n - длина массива.
    """
    for i in range(len(arr)):      # O(n) - проход по всем элементам
        if arr[i] == target:       # O(1) - сравнение
            return i               # O(1) - возврат индекса
    return -1                      # O(1) - если не найден
    # Общая сложность: O(n)


def binary_search(arr, target):
    """
    Бинарный поиск элемента в отсортированном массиве.
    Возвращает индекс target или -1, если не найден.
    Сложность: O(log n), где n - длина массива.
    """
    left = 0                       # O(1) - инициализация
    right = len(arr) - 1           # O(1) - инициализация
    while left <= right:           # O(log n) - деление диапазона
        mid = (left + right) // 2  # O(1) - вычисление середины
        if arr[mid] == target:     # O(1) - сравнение
            return mid             # O(1) - возврат индекса
        elif arr[mid] < target:    # O(1) - сравнение
            left = mid + 1         # O(1) - сдвиг границы
        else:
            right = mid - 1        # O(1) - сдвиг границы
    return -1                      # O(1) - если не найден
    # Общая сложность: O(log n)


sizes = [1000, 2000, 5000, 10000, 50000, 100000, 500000, 1000000]


def generate_test_data(sizes):
    """
    Генерирует отсортированные массивы заданных размеров и целевые элементы.
    Возвращает словарь: {size: {'array': [...], 'targets': {...}}}
    Сложность: O(k*n), где k - количество размеров, n - размер массива.
    """
    data = {}
    for size in sizes:                # O(k)
        arr = list(range(size))       # O(n)
        targets = {
            'first': arr[0],              # O(1)
            'middle': arr[size // 2],    # O(1)
            'last': arr[-1],             # O(1)
            'absent': -1                 # O(1)
        }
        data[size] = {'array': arr, 'targets': targets}  # O(1)
    return data                       # O(1)
# Общая сложность: O(k*n)


test_data = generate_test_data(sizes)


def measure_time(search_func, arr, target, repeat=10):
    times = []
    for _ in range(repeat):
        t = timeit.timeit(lambda: search_func(arr, target), number=1)
        times.append(t * 1000)
    return sum(times) / len(times)


results = {
    'linear_search': {},
    'binary_search': {}
}
for size, info in test_data.items():
    arr = info['array']
    targets = info['targets']
    results['linear_search'][size] = {}
    results['binary_search'][size] = {}
    for key, target in targets.items():
        results['linear_search'][size][key] = measure_time(
            linear_search, arr, target)
        results['binary_search'][size][key] = measure_time(
            binary_search, arr, target)


def plot_results(results, sizes):
    plt.figure(figsize=(12, 6))
    for alg in ['linear_search', 'binary_search']:
        y = [results[alg][size]['last'] for size in sizes]
        plt.plot(sizes, y, marker='o', label=alg)
    plt.xlabel('Размер массива')
    plt.ylabel('Время (мс)')
    plt.title('Время поиска (последний элемент)')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_complexity_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 6))
    for alg in ['linear_search', 'binary_search']:
        y = [results[alg][size]['last'] for size in sizes]
        plt.plot(sizes, y, marker='o', label=alg)
    plt.xlabel('Размер массива')
    plt.ylabel('Время (мс, log scale)')
    plt.yscale('log')
    plt.title('Время поиска (логарифмическая шкала, последний элемент)')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_complexity_plot_log.png', dpi=300, bbox_inches='tight')
    plt.show()

    # График в log-log масштабе
    plt.figure(figsize=(12, 6))
    for alg in ['linear_search', 'binary_search']:
        y = [results[alg][size]['last'] for size in sizes]
        plt.plot(sizes, y, marker='o', label=alg)
    plt.xlabel('Размер массива (log scale)')
    plt.ylabel('Время (мс, log scale)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Время поиска (log-log scale, последний элемент)')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_complexity_plot_log_log.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_results(results, sizes)


def print_table(alg_name, results, sizes, keys):
    print(f"\nТаблица результатов для {alg_name}:")
    header = "Размер ".ljust(10) + "| " + \
        " | ".join([k.ljust(10) for k in keys])
    print(header)
    print("-" * len(header))
    for size in sizes:
        row = str(size).ljust(10) + "| "
        row += " | ".join([f"{results[alg_name][size][key]:10.3f}" for key in keys])
        print(row)


element_keys = ['first', 'middle', 'last', 'absent']
print_table('linear_search', results, sizes, element_keys)
print_table('binary_search', results, sizes, element_keys)

print("\n--- Анализ сложности ---")
print("Линейный поиск (linear_search): теоретически O(n), время растет линейно с размером массива.\n"
      "Практически: время поиска первого элемента минимально, последнего/отсутствующего — максимально, график близок к прямой.\n"
      "Для последнего элемента требуется n сравнений.")
print("\nБинарный поиск (binary_search): теоретически O(log n), время растет медленно, логарифмически.\n"
      "Практически: время почти не зависит от позиции элемента, график близок к логарифмической кривой.\n"
      "Для последнего элемента требуется log(n) сравнений.")
