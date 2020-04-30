.. _guide_monitor:

Monitor
=======

.. currentmodule:: lab.monit

Iterators & Enumerators
-----------------------

You can use :func:`iterate` and :func:`enum` with any iterable
object. In this example we use a PyTorch ``DataLoader``.

.. code-block:: python

    # Create a data loader for illustration
    import time
    
    import torch
    from torchvision import datasets, transforms
    
    import lab
    from lab import logger, monit
    
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(lab.get_data_path(),
                           train=False,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=32, shuffle=True)

.. code-block:: python

    for data, target in monit.iterate("Test", test_loader):
        time.sleep(0.01)



.. raw:: html

    <pre>Test<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	5,806.34ms</span>
    </pre>


.. code-block:: python

    for i, (data, target) in monit.enum("Test", test_loader):
        time.sleep(0.01)



.. raw:: html

    <pre>Test<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	5,847.01ms</span>
    </pre>


Sections
--------

Sections let you monitor time taken for different tasks and also helps
keep the code clean by separating different blocks of code.

.. code-block:: python

    with monit.section("Load data"):
        # code to load data
        time.sleep(2)



.. raw:: html

    <pre>Load data<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	2,003.64ms</span>
    </pre>


.. code-block:: python

    with monit.section("Load saved model"):
        time.sleep(1)
        monit.fail()



.. raw:: html

    <pre>Load saved model<span style="color: #E75C58">...[FAIL]</span><span style="color: #208FFB">	1,006.66ms</span>
    </pre>


You can also show progress while a section is running

.. code-block:: python

    with monit.section("Train", total_steps=100):
        for i in range(100):
            time.sleep(0.1)
            # Multiple training steps in the inner loop
            logger.progress(i)
