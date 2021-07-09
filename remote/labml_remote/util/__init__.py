from pathlib import Path
from typing import Dict


def template(file: Path, replace: Dict[str, str]):
    with open(str(file), 'r') as f:
        content = f.read()
        for k, v in replace.items():
            content = content.replace(f'%%{k.upper()}%%', v)

    return content


def get_env_vars(env_vars: Dict[str, str]):
    if not env_vars:
        return ''

    exports = [f'export {k}={v}' for k, v in env_vars.items()]
    return '\n'.join(exports)