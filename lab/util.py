import io

import numpy as np
from matplotlib import pyplot


def overlay_image_green(result: np.ndarray,
                        base: np.ndarray,
                        overlay: np.ndarray,
                        base_factor: float):
    """
    #### Overlays a map on an image
    """
    result[:, :, 0] = base * base_factor
    result[:, :, 1] = base * base_factor
    result[:, :, 2] = base * base_factor
    result[:, :, 1] += (1 - base_factor) * 255 * overlay / (np.max(overlay) - np.min(overlay))


def create_png(frame: np.ndarray):
    """
    #### Create a PNG from a numpy array.
    """
    png = io.BytesIO()
    pyplot.imsave(png, frame, format='png', cmap='gray')
    return png
