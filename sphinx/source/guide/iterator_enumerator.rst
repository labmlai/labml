.. _guide_iterators_enumerators:

Iterators and Enumerators
=========================

.. currentmodule:: lab.logger

You can use :meth:`iterate` and :meth:`enum` with any iterable
object. In this example we use a PyTorch ``DataLoader``.

.. code-block:: python

    # Create a data loader for illustration
    import time
    
    import torch
    from torchvision import datasets, transforms
    
    from lab import logger
    
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(logger.get_data_path(),
                           train=False,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=32, shuffle=True)

.. code-block:: python

    for data, target in logger.iterate("Test", test_loader):
        time.sleep(0.01)



.. raw:: html

    <pre>Test<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	6,757.82ms</span>
    </pre>


.. code-block:: python

    for i, (data, target) in logger.enum("Test", test_loader):
        time.sleep(0.01)



.. raw:: html

    <pre>Test<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	6,772.50ms</span>
    </pre>

