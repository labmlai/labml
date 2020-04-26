import matplotlib.pyplot as plt
import seaborn as sns

from lab.internal.analytics import Analyzer

sns.set_context('notebook')
sns.set_style('white')


def _test_tensorboard(analyzer: Analyzer):
    Histograms, Scalars = analyzer.get_indicators('Histogram', 'Scalar')
    analyzer.tb.load()
    _, ax = plt.subplots(figsize=(18, 9))
    analyzer.tb.render_scalar(Scalars.train_loss, ax, sns.color_palette()[0])
    plt.show()

    _, ax = plt.subplots(figsize=(18, 9))
    analyzer.tb.render_histogram(Histograms.train_loss, ax, sns.color_palette()[0])
    plt.show()


if __name__ == '__main__':
    a = Analyzer('mnist_loop', '9070f0ba756511ea8c99acde48001122', False)
    _test_tensorboard(a)
