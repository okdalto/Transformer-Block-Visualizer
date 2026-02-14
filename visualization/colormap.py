import numpy as np


def matrix_to_colors(matrix: np.ndarray, alpha: float = 1.0,
                     vmax: float = None) -> np.ndarray:
    """Convert a matrix of values to RGBA colors. Returns (rows, cols, 4).
    If vmax is provided, use symmetric range [-vmax, vmax] for normalization.
    Otherwise, compute from matrix."""
    if vmax is None:
        vmax = max(abs(matrix.max()), abs(matrix.min()), 0.01)
    vmin = -vmax

    rows, cols = matrix.shape
    colors = np.zeros((rows, cols, 4), dtype=np.float32)

    # Vectorized colormap
    t = np.clip((matrix - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    mask_low = t < 0.5
    s_low = t * 2.0
    s_high = (t - 0.5) * 2.0

    colors[:, :, 0] = np.where(mask_low, s_low, 1.0)
    colors[:, :, 1] = np.where(mask_low, s_low, 1.0 - s_high)
    colors[:, :, 2] = np.where(mask_low, 1.0, 1.0 - s_high)
    colors[:, :, 3] = alpha

    return colors
