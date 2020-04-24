.. _guide_sections:

Sections
========

Sections let you monitor time taken for different tasks and also helps
*keep the code clean* by separating different blocks of code.

.. code-block:: python

    import time
    
    from lab import logger

.. code-block:: python

    with logger.section("Load data"):
        # code to load data
        time.sleep(2)



.. raw:: html

    <pre>Load data<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	2,004.50ms</span>
    </pre>


.. code-block:: python

    with logger.section("Load saved model"):
        time.sleep(1)
        logger.set_successful(False)



.. raw:: html

    <pre>Load saved model<span style="color: #E75C58">...[FAIL]</span><span style="color: #208FFB">	1,010.32ms</span>
    </pre>


You can also show progress while a section is running

.. code-block:: python

    with logger.section("Train", total_steps=100):
        for i in range(100):
            time.sleep(0.1)
            # Multiple training steps in the inner loop
            logger.progress(i)



.. raw:: html

    <pre>Train<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	10,570.79ms</span>
    </pre>

