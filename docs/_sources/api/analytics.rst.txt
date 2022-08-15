Analytics
=========

`Here is a tutorial on Google Colab that shows how to use the analytics module <https://colab.research.google.com/github/labmlai/labml/blob/master/guides/analytics.ipynb>`_

.. image:: https://raw.githubusercontent.com/vpj/lab/master/images/analytics.png
 :width: 400
 :alt: labml analytics

.. automodule:: labml.analytics

    Probing
    -------

    .. autoclass:: ModelProbe

        .. autoproperty:: parameters

        .. autoproperty:: forward_input

        .. autoproperty:: forward_output

        .. autoproperty:: backward_input

        .. autoproperty:: backward_output

    .. autoclass:: ValueCollection

        .. automethod:: get_value

        .. automethod:: get_list

        .. automethod:: get_dict

        .. automethod:: deep

    .. autoclass:: DeepValueCollection

        .. automethod:: get_value

        .. automethod:: get_list

        .. automethod:: get_dict

    Get data
    --------

    .. autofunction:: runs

    .. autofunction:: get_run

    .. autofunction:: set_preferred_db

    .. autofunction:: indicator_data

    .. autofunction:: artifact_data

    .. autoclass:: IndicatorCollection

    Plot
    ----

    .. autofunction:: distribution

    .. autofunction:: scatter

    .. autofunction:: binned_heatmap

    .. autofunction:: histogram
