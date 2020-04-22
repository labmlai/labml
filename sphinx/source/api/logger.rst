Logger
======

.. automodule:: lab.logger

    .. autofunction:: log

    .. autofunction:: add_indicator

    .. autofunction:: add_artifact

    .. autofunction:: store

    .. autofunction:: write

    .. autofunction:: new_line

    .. autofunction:: set_global_step

    .. autofunction:: add_global_step

    .. autofunction:: get_global_step

    .. autofunction:: iterate

    .. autofunction:: enum

    .. autofunction:: section

    .. autofunction:: progress

    .. autofunction:: set_successful

    .. autofunction:: loop

    .. autofunction:: finish_loop

    .. autofunction:: info

    .. autofunction:: get_data_path

    .. autofunction:: save_numpy

Colors
------

.. automodule:: lab.logger.colors

    .. autoclass:: Text

    .. autoclass:: Style

    .. autoclass:: Color

    .. autoclass:: StyleCode


Artifacts
---------

.. automodule:: lab.logger.artifacts

    .. autoclass:: Image

    .. autoclass:: Text

    .. autoclass:: IndexedText

    .. autoclass:: Artifact

Indicators
----------

.. automodule:: lab.logger.indicators

    .. autoclass:: Scalar

    .. autoclass:: Histogram

    .. autoclass:: Queue

    .. autoclass:: IndexedScalar

    .. autoclass:: Indicator

Iterators and Enumerators
-------------------------

.. automodule:: lab.logger.iterator

    .. autoclass:: Iterator

Sections
--------

.. automodule:: lab.logger.sections

    .. autoclass:: Section

        .. automethod:: get_estimated_time

        .. automethod:: progress

Utils
-----

.. automodule:: lab.logger.util

    .. automodule:: lab.logger.util.pytorch

        .. autofunction:: add_model_indicators

        .. autofunction:: store_model_indicators
