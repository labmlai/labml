import re
from pathlib import Path

from lab import logger, monit
from lab.logger import Text


def fix_file(path: Path):
    with open(str(path), 'r') as f:
        code = f.read()

    # Change code blocks
    code = re.sub(r'\n.. code:: ipython3\n',
                  r'\n.. code-block:: python\n',
                  code)

    # Change references
    pat = re.compile(r"""
     ``:                 # Start of a reference
     (?P<type>[^:]*)     # Type of reference
     :                   # colon
     (?P<name>[^`]*)     # Name of the reference
     ``                  # End
    """, re.VERBOSE)

    code = pat.sub(r':\g<type>:`\g<name>`', code)

    # Change directives such as current module
    pat = re.compile(r"""
     \n``\.\.\s          # Start of current module
     (?P<type>[^:]*)     # Type of reference
     ::\s                # colon
     (?P<name>[^`]*)     # Name of the reference
     ``                  # End
    """, re.VERBOSE)

    # Change links
    code = pat.sub(r'\n.. \g<type>:: \g<name>', code)

    pat = re.compile(r"""
     ^``\.\.\s           # Start of current module
     (?P<name>[^:]*)    # Type of reference
     :``                 # colon
    """, re.VERBOSE)

    code = pat.sub(r'.. \g<name>:', code)

    with open(str(path), 'w') as f:
        f.write(code)


def fix_folder(path: Path):
    logger.log(['Path: ', (path, Text.value)])

    for f in path.iterdir():
        if f.suffix == '.rst':
            with monit.section(str(f.name)):
                fix_file(f)


def run():
    from lab.internal.lab import lab_singleton

    lab_singleton().set_path(__file__)
    guide = Path(lab_singleton().path) / 'sphinx' / 'source' / 'guide'

    fix_folder(guide)


if __name__ == '__main__':
    run()
