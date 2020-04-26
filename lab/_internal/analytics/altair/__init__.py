from typing import List

import altair as alt

TABLEAU_10 = ['#4E79A7',
              '#F28E2C',
              '#E15759',
              '#76B7B2',
              '#59A14F',
              '#EDC949',
              '#AF7AA1',
              '#FF9DA7',
              '#9C755F',
              '#BAB0AB']


class AltairAnalytics:
    def __data_to_table(self, data):
        table = []

        for i in range(data.shape[0]):
            row = {'step': data[i, 0]}
            for j in range(1, 10):
                row[f"v{j}"] = data[i, j]
            table.append(row)

        return alt.Data(values=table)

    def __render_density(self, table: alt.Data,
                         name: str,
                         line_color: str,
                         range_color: str,
                         levels: int,
                         alpha: float,
                         selection: alt.Selection = None,
                         x_scale: alt.Scale = alt.Undefined,
                         y_scale: alt.Scale = alt.Undefined) -> alt.Chart:
        areas: List[alt.Chart] = []
        for i in range(1, levels):
            y = f"v{5 - i}:Q"
            y2 = f"v{5 + i}:Q"

            areas.append(
                alt.Chart(table)
                    .mark_area(opacity=alpha ** i)
                    .encode(x=alt.X('step:Q', scale=x_scale),
                            y=alt.Y(y, scale=y_scale),
                            y2=alt.Y2(y2),
                            color=alt.value(range_color)
                            )
            )

        line: alt.Chart = (
            alt.Chart(table)
                .mark_line()
                .encode(x=alt.X('step:Q', scale=x_scale),
                        y=alt.Y("v5:Q", scale=y_scale, title=name),
                        color=alt.value(line_color)
                        )
        )
        if selection is not None:
            line = line.add_selection(selection)

        areas_sum = None
        for a in areas:
            if areas_sum is None:
                areas_sum = a
            else:
                areas_sum += a

        if areas_sum is None:
            return line
        else:
            return areas_sum + line

    def render_density(self, data, *,
                       name: str,
                       line_color: str,
                       range_color: str,
                       levels: int,
                       alpha: float,
                       height: int,
                       width: int):
        table = self.__data_to_table(data)

        chart = self.__render_density(table,
                                      name=name,
                                      line_color=line_color,
                                      range_color=range_color,
                                      levels=levels,
                                      alpha=alpha)
        chart = chart.properties(width=width, height=height)

        return chart

    def render_density_minimap(self, data, *,
                               name: str,
                               line_color: str,
                               range_color: str,
                               levels=5,
                               alpha=0.6,
                               height: int,
                               width: int,
                               height_minimap: int):
        table = self.__data_to_table(data)

        zoom = alt.selection_interval(encodings=["x", "y"])

        minimap = self.__render_density(table,
                                        name=name,
                                        line_color=line_color,
                                        range_color=range_color,
                                        levels=levels,
                                        alpha=alpha,
                                        selection=zoom)

        detail = self.__render_density(table,
                                       name=name,
                                       line_color=line_color,
                                       range_color=range_color,
                                       levels=levels,
                                       alpha=alpha,
                                       x_scale=alt.Scale(domain={'selection': zoom.name,
                                                                 "encoding": "x"}),
                                       y_scale=alt.Scale(domain={'selection': zoom.name,
                                                                 "encoding": "y"}))
        minimap = minimap.properties(width=width, height=height_minimap)
        detail = detail.properties(width=width, height=height)

        return detail & minimap

    def render_density_minimap_multiple(self, datas, *,
                                        names: List[str],
                                        levels=5,
                                        alpha=0.6,
                                        height: int,
                                        width: int,
                                        height_minimap: int):
        tables = [self.__data_to_table(d) for d in datas]

        zoom = alt.selection_interval(encodings=["x", "y"])

        minimaps = None
        for i, t in enumerate(tables):
            z = zoom if i == 0 else None
            minimap = self.__render_density(t,
                                            name=names[0],
                                            line_color=TABLEAU_10[i],
                                            range_color=TABLEAU_10[i],
                                            levels=levels,
                                            alpha=alpha,
                                            selection=z)
            if minimaps is None:
                minimaps = minimap
            else:
                minimaps += minimap

        details = None
        for i, t in enumerate(tables):
            detail = self.__render_density(t,
                                           name=names[i],
                                           line_color=TABLEAU_10[i],
                                           range_color=TABLEAU_10[i],
                                           levels=levels,
                                           alpha=alpha,
                                           x_scale=alt.Scale(domain={'selection': zoom.name,
                                                                     "encoding": "x"}),
                                           y_scale=alt.Scale(domain={'selection': zoom.name,
                                                                     "encoding": "y"}))
            if details is None:
                details = detail
            else:
                details += detail

        minimaps = minimaps.properties(width=width, height=height_minimap)
        details = details.properties(width=width, height=height)

        return details & minimaps
