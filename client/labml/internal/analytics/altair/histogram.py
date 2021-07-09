from typing import Optional

import altair as alt
import numpy as np


def data_to_table(data, low: Optional[float], high: Optional[float]):
    table = []

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    data: np.ndarray = data.ravel()

    for i in range(data.shape[0]):
        v = data[i]
        if low is not None and v < low:
            continue
        if high is not None and v > high:
            continue
        table.append({'value': v})

    return alt.Data(values=table)


def render(table: alt.Data, *,
           height: int,
           width: int,
           height_minimap: int):
    brush = alt.selection_interval(encodings=['x'])

    base = alt.Chart(table).mark_bar().encode(
        y='count():Q'
    )
    detail = base.encode(
        alt.X('value:Q',
              bin=alt.Bin(maxbins=30, extent=brush),
              scale=alt.Scale(domain=brush)
              )
    ).properties(
        width=width,
        height=height
    )
    minimap = base.encode(
        alt.X('value:Q', bin=alt.Bin(maxbins=30)),
    ).add_selection(brush).properties(width=width, height=height_minimap)

    return detail & minimap
