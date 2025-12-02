import numpy as np
from PIL import Image


def get_bicubic_weights(dist, a=-0.5):
    w = np.zeros((4, *dist.shape))

    d1 = 1 + dist
    d2 = dist
    d3 = 1 - dist
    d4 = 2 - dist

    w[0] = a * d1 ** 3 - 5 * a * d1 ** 2 + 8 * a * d1 - 4 * a
    w[1] = (a + 2) * d2 ** 3 - (a + 3) * d2 ** 2 + 1
    w[2] = (a + 2) * d3 ** 3 - (a + 3) * d3 ** 2 + 1
    w[3] = a * d4 ** 3 - 5 * a * d4 ** 2 + 8 * a * d4 - 4 * a

    return w

def SR_bicubic(img, new_height, new_width):
    height, width, channels = img.shape

    img_padded = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode='symmetric')

    row_indices = np.linspace(0, height, new_height, endpoint=False)
    col_indices = np.linspace(0, width, new_width, endpoint=False)

    print(row_indices)
    print(col_indices)

    row_floor = row_indices.astype(int)
    col_floor = col_indices.astype(int)

    row_frac = row_indices - row_floor
    col_frac = col_indices - col_floor

    row_floor += 2
    col_floor += 2

    # 3. Вычисляем веса для строк и столбцов
    # W_rows имеет форму (4, new_height)
    # W_cols имеет форму (4, new_width)
    W_rows = get_bicubic_weights(row_frac)
    W_cols = get_bicubic_weights(col_frac)

    # 4. Интерполяция
    # Мы будем использовать матричные операции для всех каналов сразу

    # Собираем индексы 16 соседей для каждой точки
    # Нам нужно выбрать строки: row_floor-1, row_floor, row_floor+1, row_floor+2
    row_idx = np.stack([row_floor - 1, row_floor, row_floor + 1, row_floor + 2], axis=0)
    col_idx = np.stack([col_floor - 1, col_floor, col_floor + 1, col_floor + 2], axis=0)

    # --- Интерполяция по вертикали (Rows) ---
    # Сначала сжимаем по вертикали, получая промежуточный результат
    # img_padded имеет размеры (H+4, W+4, 3)
    # Нам нужно выбрать нужные строки для каждого нового пикселя по Y

    # Формируем выборку строк. Размеры: (4, new_height, W+4, 3)
    rows_subset = img_padded[row_idx, :, :]

    # Умножаем на веса строк и суммируем по оси 0 (ось 4-х соседей)
    # W_rows: (4, new_height) -> расширяем до (4, new_height, 1, 1) для умножения
    W_rows_exp = W_rows[:, :, None, None]

    # Промежуточный итог: интерполяция по Y выполнена
    # Размер: (new_height, W+4, 3)
    intermediate = np.sum(rows_subset * W_rows_exp, axis=0)

    # --- Интерполяция по горизонтали (Cols) ---
    # Теперь работаем с intermediate и интерполируем по X

    # Выбираем нужные столбцы. Размеры: (4, new_width, new_height, 3)
    # Транспонируем intermediate для удобства индексации: (W+4, new_height, 3)
    intermediate = intermediate.transpose(1, 0, 2)

    cols_subset = intermediate[col_idx, :, :]

    # Веса столбцов: (4, new_width) -> (4, new_width, 1, 1)
    W_cols_exp = W_cols[:, :, None, None]

    # Суммируем
    result_transposed = np.sum(cols_subset * W_cols_exp, axis=0)

    # Возвращаем оси обратно: (new_width, new_height, 3) -> (new_height, new_width, 3)
    result = result_transposed.transpose(1, 0, 2)

    # 5. Обрезка значений (Clamping) и приведение типов
    # Бикубическая интерполяция может дать значения <0 или >255
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

rand_img = np.random.randint(0, 256, (5,5,3), dtype=np.uint8)
resized_img = SR_bicubic(rand_img, 10, 10)