import json
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

JS_CSS_ADDED = False


def init_inline_viz():
    html = ''

    global JS_CSS_ADDED

    if not JS_CSS_ADDED:
        html += '''<script src="https://labml.ai/cdn/charts.js"></script>'''
        html += '''<link rel="stylesheet" href="https://labml.ai/cdn/charts.css">'''
        JS_CSS_ADDED = True

    from IPython.core.display import display, HTML

    display(HTML(html))


def text_attention(attn: 'torch.Tensor', src_tokens: List[str], tgt_tokens: List[str]):
    assert len(attn.shape) == 2
    assert attn.shape[0] == len(src_tokens)
    assert attn.shape[1] == len(tgt_tokens)

    html = ''

    from uuid import uuid1
    elem_id = 'id_' + uuid1().hex

    html += f'<div id="{elem_id}"></div>'

    src = json.dumps([json.dumps(t)[1:-1] for t in src_tokens])
    tgt = json.dumps([json.dumps(t)[1:-1] for t in tgt_tokens])

    attn_map = json.dumps(attn.numpy().tolist())

    script = ''
    script += '<script>'
    # script += f"function func_{elem_id[3:]}()" + "{"
    script += f"window.chartsEmbed('{elem_id}', {src}, {tgt}, {attn_map})"
    # script += '}'
    # script += "if(window.chartsEmbed != null) {" + f'func_{elem_id[3:]}()' + "}"
    # script += "else { document.addEventListener('DOMContentLoaded', " + f'func_{elem_id[3:]}' + ')}'
    script += '</script>'

    from IPython.core.display import display, HTML

    display(HTML(html))
    display(HTML(script))


def _test():
    import torch
    text_attention(torch.Tensor([[0.]]), ['a'], ['b'])


if __name__ == '__main__':
    _test()
