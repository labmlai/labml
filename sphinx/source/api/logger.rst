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

    .. autofunction:: save_checkpoint

    .. autofunction:: info

    .. autofunction:: get_data_path

    .. autofunction:: save_numpy

Colors
------

.. automodule:: lab.logger.colors

    .. autoclass:: Text
        :members:
        :undoc-members:

    .. autoclass:: Style
        :members:
        :undoc-members:

    .. autoclass:: Color
        :members:
        :undoc-members:

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

        .. automethod:: get_estimated_time

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
