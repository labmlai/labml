Tracker
=======

Here you specify indicators and the logger stores them temporarily and
write in batches. It can aggregate and write them as means or
histograms.

.. code-block:: python

    import time
    
    import numpy as np
    
    from lab import loop, tracker, logger
    
    # dummy train function
    def train():
        return np.random.randint(100)
    
    # Reset global step because we incremented in previous loop
    loop.set_global_step(0)

This stores all the loss values and writes the logs the mean on every
tenth iteration. Console output line is replaced until
:func:`lab.logger.log` is called.

.. code-block:: python

    for i in range(1, 401):
        loop.add_global_step()
        loss = train()
        tracker.add(loss=loss)
        if i % 10 == 0:
            tracker.save()
        if i % 100 == 0:
            logger.log()
        time.sleep(0.02)



.. raw:: html

    <pre>
    
    
    </pre>


Indicator settings
------------------

.. code-block:: python

    # dummy train function
    def train2(idx):
        return idx, 10, np.random.randint(100)
    
    # Reset global step because we incremented in previous loop
    loop.set_global_step(0)

Histogram indicators will log a histogram of data. Queue will store data
in a ``deque`` of size ``queue_size``, and log histograms. Both of these
will log the means too. And if ``is_print`` is ``True`` it will print
the mean.

queue size of ``10`` and the values are printed to the console

.. code-block:: python

    tracker.set_queue('reward', 10, True)

By default values are not printed to console; i.e. ``is_print`` defaults
to ``False``.

.. code-block:: python

    tracker.set_scalar('policy')

Settings ``is_print`` to ``True`` will print the mean value of histogram
to console

.. code-block:: python

    tracker.set_histogram('value', True)

.. code-block:: python

    for i in range(1, 400):
        loop.add_global_step()
        reward, policy, value = train2(i)
        tracker.add(reward=reward, policy=policy, value=value, loss=1.)
        if i % 10 == 0:
            tracker.save()
        if i % 100 == 0:
            logger.log()



.. raw:: html

    <pre>
    
    
    <strong><span style="color: #DDB62B">     390:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 385.500</strong> value: <strong> 56.6000</strong></pre>


