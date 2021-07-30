Tracker
=======

`Here is a tutorial on Google Colab that shows how to use the tracker module <https://colab.research.google.com/github/labmlai/labml/blob/master/guides/tracker.ipynb>`_

.. automodule:: labml.tracker

    Track
    -----

    .. autofunction:: save

    .. autofunction:: add


    Step
    ----
    .. autofunction:: set_global_step

    .. autofunction:: add_global_step

    .. autofunction:: get_global_step


    Setup
    -----

    .. autofunction:: set_queue

    .. autofunction:: set_histogram

    .. autofunction:: set_scalar

    .. autofunction:: set_indexed_scalar

    .. warning::

        Artificact setup functions
        :func:`labml.tracker.set_image`,
        :func:`labml.tracker.set_text`,
        :func:`labml.tracker.set_tensor`, and
        :func:`labml.tracker.set_indexed_text` are still experimental.

    .. autofunction:: set_image

    .. autofunction:: set_text

    .. autofunction:: set_tensor

    .. autofunction:: set_indexed_text

    Namespaces
    ----------

    .. autofunction:: namespace


    Helpers
    -------

    .. autofunction:: reset

    .. autofunction:: new_line
