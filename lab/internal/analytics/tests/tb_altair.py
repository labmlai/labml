from lab.internal.analytics import Analyzer


def _test_tensorboard(analyzer: Analyzer):
    Histograms, Scalars = analyzer.get_indicators('Histogram', 'Scalar')
    analyzer.tb.load()
    # analyzer.tb.render_scalar(Scalars.train_loss, sns.color_palette()[0])
    import altair as alt

    alt.renderers.enable('altair_viewer')

    chart = analyzer.tb.render_histogram(Histograms.train_loss, None)
    chart = chart.properties(width=800, height=400, title="Histogram")
    chart = chart.interactive()
    chart.show()

    chart = analyzer.tb.render_scalar(Scalars.train_loss, None)
    chart = chart.properties(width=800, height=400, title="Scalar")
    chart = chart.interactive()
    chart.show()


if __name__ == '__main__':
    a = Analyzer('mnist_loop', '9070f0ba756511ea8c99acde48001122')
    _test_tensorboard(a)
