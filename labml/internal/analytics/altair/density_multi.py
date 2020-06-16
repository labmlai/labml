from typing import List

import altair as alt
from labml.internal.analytics.altair.utils import TABLEAU_10

try:
    import torch
except ImportError:
    torch = None


def data_to_table(data, step):
    table = []

    if torch is not None and isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    for i in range(data.shape[0]):
        if len(data.shape) == 2:
            m = data.shape[1] // 2
            row = {'step': data[i, 0], 'v5': data[i, m]}
            for j in range(1, min(m, 5)):
                row[f"v{5 - j}"] = data[i, m - j]
                row[f"v{5 + j}"] = data[i, m + j]
        elif step is not None:
            row = {'step': step[i],
                   'v5': data[i]}
        else:
            row = {'step': i,
                   'v5': data[i]}
        table.append(row)

    return alt.Data(values=table)


def _render_density(table: alt.Data, *,
                    name: str,
                    x_name: str,
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
            .encode(x=alt.X('step:Q', scale=x_scale, title=x_name),
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


def render(tables: List[alt.Data], *,
           names: List[str],
           is_independent: bool,
           levels=5,
           alpha=0.6,
           height: int,
           width: int,
           height_minimap: int):
    zoom = alt.selection_interval(encodings=["x", "y"])

    minimaps = None
    for i, t in enumerate(tables):
        z = zoom if i == 0 else None
        minimap = _render_density(t,
                                  name='',
                                  x_name='',
                                  line_color=TABLEAU_10[i % 10],
                                  range_color=TABLEAU_10[i % 10],
                                  levels=levels,
                                  alpha=alpha,
                                  selection=z)
        if minimaps is None:
            minimaps = minimap
        else:
            minimaps += minimap

    details = None
    for i, t in enumerate(tables):
        detail = _render_density(t,
                                 name=names[i] if len(names) == 1 else '',
                                 x_name='Step',
                                 line_color=TABLEAU_10[i % 10],
                                 range_color=TABLEAU_10[i % 10],
                                 levels=levels,
                                 alpha=alpha,
                                 x_scale=alt.Scale(domain={'selection': zoom.name,
                                                           "encoding": "x"}),
                                 y_scale=alt.Scale(domain={'selection': zoom.name,
                                                           "encoding": "y"}))
        if details is None:
            details = detail
        elif is_independent:
            details = alt.layer(details, detail).resolve_scale(y='independent')
        else:
            details += detail

    colors = alt.Data(values=[{'idx': i, 'name': name} for i, name in enumerate(names)])
    legend = alt.Chart(colors).mark_rect().encode(
        alt.Y('name:N',
              sort=alt.EncodingSortField(field='i', order='ascending'),
              axis=alt.Axis(
                  orient='right',
                  titleX=7,
                  titleY=-2,
                  titleAlign='left',
                  titleAngle=0
              ),
              title=''),
        alt.Color('idx:O',
                  scale=alt.Scale(domain=list(range(len(tables))),
                                  range=TABLEAU_10),
                  legend=None))

    minimaps = minimaps.properties(width=width, height=height_minimap)
    details = details.properties(width=width, height=height)

    if len(names) == 1:
        return details & minimaps
    else:
        return (details & minimaps) | legend
