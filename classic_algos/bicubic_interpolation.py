import numpy as np


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

def SR_bicubic(img, new_height, new_width, output_dtype=None, preserve_range=False):
    height, width, channels = img.shape

    img_padded = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode='symmetric')

    # positions in old image
    row_indices = np.linspace(0, height, new_height, endpoint=False)
    col_indices = np.linspace(0, width, new_width, endpoint=False)
    row_floor = row_indices.astype(int)
    col_floor = col_indices.astype(int)
    row_frac = row_indices - row_floor
    col_frac = col_indices - col_floor

    #padding
    row_floor += 2
    col_floor += 2

    W_rows = get_bicubic_weights(row_frac)
    W_cols = get_bicubic_weights(col_frac)

    # choose indices of the 4 neighboring pixels (4 rows and 4 columns)
    row_idx = np.stack([row_floor - 1, row_floor, row_floor + 1, row_floor + 2], axis=0)
    col_idx = np.stack([col_floor - 1, col_floor, col_floor + 1, col_floor + 2], axis=0)

    # (4, new_height, W+4, C) scaled matrix.
    rows_subset = img_padded[row_idx, :, :]

    # W_rows: (4, new_height) -> (4, new_height, 1, 1)
    W_rows_exp = W_rows[:, :, None, None]

    # squeeze these 4 rows into one with weighted sum -> (new_height, W+4, C)
    intermediate = np.sum(rows_subset * W_rows_exp, axis=0)

    # transpose matrix and repeat for columns
    intermediate = intermediate.transpose(1, 0, 2)
    cols_subset = intermediate[col_idx, :, :]
    W_cols_exp = W_cols[:, :, None, None]
    result_transposed = np.sum(cols_subset * W_cols_exp, axis=0)
    result = result_transposed.transpose(1, 0, 2)

    #DL
    if preserve_range or (output_dtype is not None and output_dtype != np.uint8):
        if output_dtype is None:
            output_dtype = np.float32 if np.issubdtype(img.dtype, np.floating) else img.dtype
        return result.astype(output_dtype)

    #standard
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result