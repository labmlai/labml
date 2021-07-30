Tracker
=======

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

        :func:`labml.tracker.set_image`,
        :func:`labml.tracker.set_text`, and
        :func:`labml.tracker.set_indexed_text` are still experimental.

    .. autofunction:: set_image

    .. autofunction:: set_text

    .. autofunction:: set_indexed_text

    Namespaces
    ----------

    .. autofunction:: namespace


    Helpers
    -------

    .. autofunction:: reset

    .. autofunction:: new_line
