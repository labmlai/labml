Loop
====

.. currentmodule:: lab.logger

This can be used for the training loop.
The :meth:`loop` keeps track of the time taken and time remaining for the loop.
You can use :ref:`guide_sections`, :ref:`guide_iterators_enumerators` within loop.

:meth:`write` outputs the current status along with global step.


.. code-block:: python

    for step in logger.loop(range(0, 400)):
        logger.write()


.. raw:: html

    <pre><strong><span style="color: #DDB62B">     399:  </span></strong>  <span style="color: #208FFB">1ms</span><span style="color: #D160C4">  0:00m/  0:00m  </span></pre>


The global step is used for logging to the screen, TensorBoard and when logging analytics to SQLite. You can manually set the global step. Here we will reset it.


.. code-block:: python

    logger.set_global_step(0)

You can manually increment global step too.


.. code-block:: python

    for step in logger.loop(range(0, 400)):
        logger.add_global_step(5)
        logger.write()

.. raw:: html

    <pre><strong><span style="color: #DDB62B">   2,000:  </span></strong>  <span style="color: #208FFB">1ms</span><span style="color: #D160C4">  0:00m/  0:00m  </span></pre>
