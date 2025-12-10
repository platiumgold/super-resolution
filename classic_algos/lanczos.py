import numpy as np

def sinc(x):
    return np.where(x == 0, 1.0, np.sin(np.pi * x) / (np.pi * x))


def get_lanczos_weights(dist, a=3):
    weights = []
    for k in range(-a + 1, a + 1):
        delta = k - dist

        w_k = sinc(delta) * sinc(delta / a)
        weights.append(w_k)

    weights = np.array(weights)

    weights = weights / np.sum(weights, axis=0)

    return weights


def SR_lanczos(img, new_height, new_width, a=3, output_dtype=None, preserve_range=False):
    height, width, channels = img.shape

    img_padded = np.pad(img, ((a, a), (a, a), (0, 0)), mode='symmetric')

    # positions in old image
    row_indices = np.linspace(0, height, new_height, endpoint=False)
    col_indices = np.linspace(0, width, new_width, endpoint=False)
    row_floor = row_indices.astype(int)
    col_floor = col_indices.astype(int)
    row_frac = row_indices - row_floor
    col_frac = col_indices - col_floor

    #padding
    row_floor += a
    col_floor += a

    W_rows = get_lanczos_weights(row_frac, a=a)
    W_cols = get_lanczos_weights(col_frac, a=a)

    # choose indices of the a neighboring pixels
    offsets = np.arange(-a + 1, a + 1)
    row_idx = np.stack([row_floor + k for k in offsets], axis=0)
    col_idx = np.stack([col_floor + k for k in offsets], axis=0)

    # scaled matrix with neighboring rows: (6, new_height, W+2a, 3)
    rows_subset = img_padded[row_idx, :, :]

    W_rows_exp = W_rows[:, :, None, None]

    # squeeze these 4 rows into one with weighted sum
    # (new_height, W+2a, 3)
    intermediate = np.sum(rows_subset * W_rows_exp, axis=0)

    # transpose matrix and repeat for columns
    intermediate = intermediate.transpose(1, 0, 2)
    cols_subset = intermediate[col_idx, :, :]
    W_cols_exp = W_cols[:, :, None, None]
    result_transposed = np.sum(cols_subset * W_cols_exp, axis=0)

    result = result_transposed.transpose(1, 0, 2)

    if preserve_range or (output_dtype is not None and output_dtype != np.uint8):
        if output_dtype is None:
            output_dtype = np.float32 if np.issubdtype(img.dtype, np.floating) else img.dtype
        return result.astype(output_dtype)

    return np.clip(result, 0, 255).astype(np.uint8)