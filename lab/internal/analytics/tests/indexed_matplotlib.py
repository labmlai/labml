import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lab.internal.analytics import Analyzer

sns.set_context('notebook')
sns.set_style('white')


def _test_simple(a: Analyzer):
    Histograms, Scalars = a.get_indicators('Histogram', 'Scalar')
    _, ax = plt.subplots(figsize=(18, 9))

    a.sqlite.render_scalar(Scalars.train_loss, ax, sns.color_palette()[0])
    plt.show()

    # Tensorboard tensor histogram parsing is not implemented
    # a.tb.load()
    # _, ax = plt.subplots(figsize=(18, 9))
    # a.tb.render_histogram(Histograms.train_loss, ax, sns.color_palette()[0])
    # plt.show()


def _test_indexed(a: Analyzer):
    Histograms, Scalars = a.get_indicators('Histogram', 'Scalar')
    cur = a.sqlite.conn.execute(
        f'SELECT * from indexed_scalars WHERE indicator = "{a.sqlite.get_key(Scalars.test_sample_loss)}"')
    res = [c for c in cur]
    steps = list({r[1] for r in res})
    steps.sort()
    steps_idx = {s: i for i, s in enumerate(steps)}
    table = np.zeros((len(steps), 10000), np.float)
    for r in res:
        idx = steps_idx[r[1]]
        table[idx, r[2]] = r[3]
    for i in range(100):
        plt.plot(range(len(steps)), table[:, i], lw=1, alpha=0.5, )
    plt.show()
    print('done')


if __name__ == '__main__':
    a = Analyzer('mnist_indexed_logs', 'e09b8b3e733211ea8056acde48001122', False)
    _test_simple(a)
    _test_indexed(a)
