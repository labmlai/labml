from typing import List

import numpy as np
import tensorflow as tf


class MatPlotLibTensorBoardAnalytics:
    @staticmethod
    def render_matrix(matrix, ax, color):
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection

        rows, cols = matrix.shape
        x_ticks = [matrix[0, i] for i in range(1, cols)]
        y_ticks = [matrix[i, 0] for i in range(1, rows)]

        max_density = 0
        for y in range(rows - 2):
            for x in range(cols - 2):
                area = (x_ticks[x + 1] - x_ticks[x]) * (y_ticks[y + 1] - y_ticks[y])
                density = matrix[y + 1, x + 1] / area
                if max_density < density:
                    max_density = density

        boxes = []
        for y in range(rows - 2):
            for x in range(cols - 2):
                width = x_ticks[x + 1] - x_ticks[x]
                height = y_ticks[y + 1] - y_ticks[y]
                area = width * height
                density = matrix[y + 1, x + 1] / area
                density = max(density, 1)
                # alpha = np.log(density) / np.log(max_density)
                alpha = density / max_density
                rect = Rectangle((x_ticks[x], y_ticks[y]), width, height,
                                 alpha=alpha, color=color)
                boxes.append(rect)

        pc = PatchCollection(boxes, match_original=True)

        ax.add_collection(pc)

        # Plot something
        _ = ax.errorbar([], [], xerr=[], yerr=[],
                        fmt='None', ecolor='k')

        return x_ticks[0], x_ticks[-1], y_ticks[0], y_ticks[-1]

    def render_tensors(self, tensors: List[any], axes: np.ndarray, color):
        assert len(axes.shape) == 2
        assert axes.shape[0] * axes.shape[1] == len(tensors)
        x_min = x_max = y_min = y_max = 0

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                idx = i * axes.shape[1] + j
                axes[i, j].set_title(f"{tensors[idx].step :,}")
                matrix = tf.make_ndarray(tensors[idx].tensor_proto)
                x1, x2, y1, y2 = self.render_matrix(matrix,
                                                    axes[i, j],
                                                    color)
                x_min = min(x_min, x1)
                x_max = max(x_max, x2)
                y_min = min(y_min, y1)
                y_max = max(y_max, y2)

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i, j].set_xlim(x_min, x_max)
                axes[i, j].set_ylim(y_min, y_max)
