from typing import List, Optional, Tuple

import altair as alt
import numpy as np

from labml.internal.analytics.altair.utils import TABLEAU_10


class AltairScatter:
    def data_to_table(self, data: np.ndarray, x_data: np.ndarray,
                      noise: Optional[Tuple[float, float]]):
        table = []

        if noise:
            nx = np.random.rand(data.shape[0]) * noise[1]
            ny = np.random.rand(data.shape[0]) * noise[0]
        else:
            nx = np.zeros(data.shape[0])
            ny = np.zeros(data.shape[0])

        for i in range(data.shape[0]):
            row = {'step': x_data[i, 0],
                   'x': x_data[i, 5] + nx[i],
                   'y': data[i, 5] + ny[i]}
            table.append(row)

        return alt.Data(values=table)

    def _scatter_chart(self, table: alt.Data, *,
                       is_ticks: bool,
                       name: str,
                       x_name: str,
                       range_color: str,
                       height: int = None,
                       width: int = None,
                       selection: alt.Selection = None,
                       x_scale: alt.Scale = alt.Undefined,
                       y_scale: alt.Scale = alt.Undefined) -> alt.Chart:
        base = alt.Chart(table)
        if selection is not None:
            base = base.add_selection(selection)

        if not is_ticks:
            scat_x_title = x_name
            scat_y_title = name
        else:
            scat_x_title = ''
            scat_y_title = ''

        scat = (base
                .mark_circle()
                .encode(x=alt.X('x:Q', scale=x_scale, title=scat_x_title),
                        y=alt.Y('y:Q', scale=y_scale, title=scat_y_title),
                        opacity='step:Q',
                        color=alt.value(range_color)
                        ))

        if is_ticks:
            tick_axis = alt.Axis(labels=False, domain=False, ticks=False)

            x_ticks = base.mark_tick().encode(
                x=alt.X('x:Q', axis=tick_axis, scale=x_scale, title=x_name),
                opacity='step:Q',
                color=alt.value(range_color)
            )

            y_ticks = alt.Chart(table).mark_tick().encode(
                y=alt.X('y:Q', axis=tick_axis, scale=y_scale, title=name),
                opacity='step:Q',
                color=alt.value(range_color)
            )

            scat = scat.properties(width=width, height=height)
            x_ticks = x_ticks.properties(width=width)
            y_ticks = y_ticks.properties(height=height)
            scat = y_ticks | (scat & x_ticks)

        return scat

    def scatter(self, datas: List[np.ndarray], x_data: np.ndarray, *,
                names: List[str],
                x_name: str,
                height: int,
                width: int,
                height_minimap: int,
                noise: Optional[Tuple[float, float]]):
        tables = [self.data_to_table(d, x_data, noise) for d in datas]

        zoom = alt.selection_interval(encodings=["x", "y"])

        minimaps = None
        for i, t in enumerate(tables):
            z = zoom if i == 0 else None
            minimap = self._scatter_chart(t,
                                          is_ticks=False,
                                          range_color=TABLEAU_10[i],
                                          name='',
                                          x_name='',
                                          selection=z)
            if minimaps is None:
                minimaps = minimap
            else:
                minimaps += minimap

        details = None
        for i, t in enumerate(tables):
            detail = self._scatter_chart(t,
                                         is_ticks=len(tables) == 1,
                                         name=names[i],
                                         x_name=x_name,
                                         range_color=TABLEAU_10[i],
                                         height=height,
                                         width=width,
                                         x_scale=alt.Scale(domain={'selection': zoom.name,
                                                                   "encoding": "x"}),
                                         y_scale=alt.Scale(domain={'selection': zoom.name,
                                                                   "encoding": "y"}))
            if details is None:
                details = detail
            else:
                details += detail

        minimaps = minimaps.properties(width=width * height_minimap / height, height=height_minimap)

        return details & minimaps
