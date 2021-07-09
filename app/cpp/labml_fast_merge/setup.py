import numpy as np
from setuptools import setup

from Cython.Build import cythonize

setup(ext_modules=cythonize("labml_fast_merge.pyx"),
      include_dirs=[np.get_include()])
