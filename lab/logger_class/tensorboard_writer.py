import numpy as np
import tensorflow as tf

import lab.logger_class.writers


def _get_histogram(values):
    """
    Get TensorBoard histogram from a numpy array.
    """

    values = np.array(values)
    hist = tf.compat.v1.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    counts, bin_edges = np.histogram(values, bins=20)
    bin_edges = bin_edges[1:]

    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    return hist


_HISTOGRAM_QUANTILES_10 = [i / 10. for i in range(11)]


def __get_zerod_histogram(x):
    x_min = np.min(x)
    x_max = np.max(x)
    if not x_min < 0 < x_max:
        _, x_e = np.histogram(x, bins=10)
        return x_e

    width = (x_max - x_min) / 10
    left = np.floor((1e-6 - x_min) / width)
    right = np.floor((1e-6 + x_max) / width)
    if left > right:
        width = -x_min / left
        x_e = [x_min + i * width for i in range(11)]
    else:
        width = x_max / right
        x_e = [x_max - i * width for i in reversed(range(11))]

    return np.array(x_e)


def _get_pair_histogram(values):
    """
    Get TensorBoard tensor heat map
    """

    x = np.array([v[0] for v in values])
    y = np.array([v[1] for v in values])
    x_e = __get_zerod_histogram(x)
    y_e = __get_zerod_histogram(y)
    # x_e = np.quantile(x, q=_HISTOGRAM_QUANTILES_10)
    # y_e = np.quantile(y, q=_HISTOGRAM_QUANTILES_10)

    a = np.zeros((12, 12), np.float32)
    for i, e in enumerate(x_e):
        a[0, i + 1] = e
    for i, e in enumerate(y_e):
        a[i + 1, 0] = e

    for v in values:
        x_i = 11
        y_i = 11
        for i, e in enumerate(x_e[1:]):
            if v[0] < e:
                x_i = i + 1
                break
        for i, e in enumerate(y_e[1:]):
            if v[1] < e:
                y_i = i + 1
                break

        a[y_i, x_i] += 1

    return tf.compat.v1.make_tensor_proto(a)


class Writer(lab.logger_class.writers.Writer):
    def __init__(self, file_writer: tf.compat.v1.summary.FileWriter):
        super().__init__()

        self.__writer = file_writer

    def write(self, *, global_step: int,
              queues,
              histograms,
              pairs,
              scalars,
              tf_summaries):
        summary = tf.compat.v1.Summary()

        for k, v in queues.items():
            if len(v) == 0:
                continue
            summary.value.add(tag=k, histo=_get_histogram(v))
            summary.value.add(tag=f"{k}_mean", simple_value=float(np.mean(v)))

        for k, v in histograms.items():
            if len(v) == 0:
                continue
            summary.value.add(tag=k, histo=_get_histogram(v))
            summary.value.add(tag=f"{k}_mean", simple_value=float(np.mean(v)))

        for k, v in pairs.items():
            if len(v) == 0:
                continue
            summary.value.add(tag=k, tensor=_get_pair_histogram(v))

        for k, v in scalars.items():
            if len(v) == 0:
                continue
            summary.value.add(tag=k, simple_value=float(np.mean(v)))

        self.__writer.add_summary(summary, global_step=global_step)

        for v in tf_summaries:
            self.__writer.add_summary(v, global_step=global_step)
