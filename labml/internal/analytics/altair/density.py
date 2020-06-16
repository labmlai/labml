from typing import List

import altair as alt

try:
    import torch
except ImportError:
    torch = None


def data_to_table(series, names, step):
    table = []

    for s in range(len(series)):
        data = series[s]
        name = names[s]

        if torch is not None and isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        for i in range(data.shape[0]):
            if len(data.shape) == 2:
                m = data.shape[1] // 2
                row = {'series': name, 'step': data[i, 0], 'v5': data[i, m]}
                for j in range(1, min(m, 5)):
                    row[f"v{5 - j}"] = data[i, m - j]
                    row[f"v{5 + j}"] = data[i, m + j]
            elif step is not None:
                row = {'series': name,
                       'step': step[i],
                       'v5': data[i]}
            else:
                row = {'series': name,
                       'step': i,
                       'v5': data[i]}
            table.append(row)

    return alt.Data(values=table)


def _render_density(table: alt.Data, *,
                    x_name: str,
                    levels: int,
                    alpha: float,
                    series_selection: alt.Selection = None,
                    selection: alt.Selection = None,
                    x_scale: alt.Scale = alt.Undefined,
                    y_scale: alt.Scale = alt.Undefined) -> alt.Chart:
    areas: List[alt.Chart] = []
    for i in range(1, levels):
        y = f"v{5 - i}:Q"
        y2 = f"v{5 + i}:Q"

        encode = dict(
            x=alt.X('step:Q', scale=x_scale),
            y=alt.Y(y, scale=y_scale),
            y2=alt.Y2(y2),
            color=alt.Color('series:N', scale=alt.Scale(scheme='tableau10'))
        )

        if series_selection:
            encode['opacity'] = alt.condition(series_selection, alt.value(alpha ** i),
                                              alt.value(0.001))

        areas.append(
            alt.Chart(table)
                .mark_area(opacity=alpha ** i)
                .encode(**encode)
        )

    encode = dict(x=alt.X('step:Q', scale=x_scale, title=x_name),
                  y=alt.Y("v5:Q", scale=y_scale, title='Value'),
                  color=alt.Color('series:N', scale=alt.Scale(scheme='tableau10')))

    if series_selection:
        encode['opacity'] = alt.condition(series_selection, alt.value(1), alt.value(0.001))

    line: alt.Chart = (
        alt.Chart(table)
            .mark_line()
            .encode(**encode)
    )
    if selection is not None:
        line = line.add_selection(selection)

    areas_sum = None
    for a in areas:
        if areas_sum is None:
            areas_sum = a
        else:
            areas_sum += a

    if areas_sum is not None:
        line = areas_sum + line

    if series_selection:
        line = line.add_selection(series_selection)

    return line


def render(table: alt.Data, *,
           levels=5,
           alpha=0.6,
           height: int,
           width: int,
           height_minimap: int):
    zoom = alt.selection_interval(encodings=["x", "y"])
    selection = alt.selection_multi(fields=['series'], bind='legend')

    minimaps = _render_density(table,
                               x_name='',
                               levels=levels,
                               alpha=alpha,
                               selection=zoom)

    details = _render_density(table,
                              x_name='Step',
                              levels=levels,
                              alpha=alpha,
                              series_selection=selection,
                              x_scale=alt.Scale(domain={'selection': zoom.name,
                                                        "encoding": "x"}),
                              y_scale=alt.Scale(domain={'selection': zoom.name,
                                                        "encoding": "y"}))

    minimaps = minimaps.properties(width=width, height=height_minimap)
    details = details.properties(width=width, height=height)

    return details & minimaps
