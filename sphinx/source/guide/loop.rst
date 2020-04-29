Loop
====

.. currentmodule::`lab.loop``

This can be used for the training loop. The :func:`loop` keeps track
of the time taken and time remaining for the loop. You can use
:ref:`guide_monitor` within loop.

:func:`lab.tracker.save` outputs the current status along with global
step.

.. code:: ipython3

    from lab import loop, tracker

.. code:: ipython3

    for step in loop.loop(range(0, 400)):
        tracker.save()



.. raw:: html

    <pre><strong><span style="color: #DDB62B">     399:  </span></strong>  <span style="color: #208FFB">1ms</span><span style="color: #D160C4">  0:00m/  0:00m  </span></pre>


The global step is used for logging to the screen, TensorBoard and when
logging analytics to SQLite. You can manually set the global step. Here
we will reset it.

.. code:: ipython3

    loop.set_global_step(0)

You can manually increment global step too.

.. code:: ipython3

    for step in loop.loop(range(0, 400)):
        loop.add_global_step(5)
        tracker.save()



.. raw:: html

    <pre><strong><span style="color: #DDB62B">   2,000:  </span></strong>  <span style="color: #208FFB">1ms</span><span style="color: #D160C4">  0:00m/  0:00m  </span></pre>


