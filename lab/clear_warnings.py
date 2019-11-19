"""
## Clears python warnings
"""

import warnings
import os

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore",
                        message="Conversion of the second argument of issubdtype"
                                " from `float` to `np.floating` is deprecated.")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
