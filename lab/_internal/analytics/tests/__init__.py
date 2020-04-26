def _test_altair():
    """
    https://altair-viz.github.io/user_guide/display_frontends.html#display-general
    """
    import altair as alt

    alt.renderers.enable('altair_viewer')

    data = alt.Data(values=[{'x': 'A', 'y': 5},
                            {'x': 'B', 'y': 3},
                            {'x': 'C', 'y': 6},
                            {'x': 'D', 'y': 7},
                            {'x': 'E', 'y': 2}])
    chart = alt.Chart(data).mark_bar().encode(
        x='x:O',  # specify ordinal data
        y='y:Q',  # specify quantitative data
    ).interactive()

    chart.show()


if __name__ == '__main__':
    _test_altair()
