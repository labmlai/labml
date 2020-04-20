Logger
======

.. currentmodule:: lab.logger

.. code-block:: python

    from lab import logger
    from lab.logger.colors import Text, Color

You can log to the screen with :meth:`log`.

Logging with colors
-------------------

.. code-block:: python

    logger.log("Colors are missing when views on github", Text.highlight)

.. raw:: html

    <pre><strong><span style="color: #DDB62B">Colors are missing when views on github</span></strong></pre>


You can use predefined styles


.. code-block:: python

    logger.log([
        ('Styles ', Text.heading),
        ('Danger ', Text.danger),
        ('Warning ', Text.warning),
        ('Meta ', Text.meta),
        ('Key ', Text.key),
        ('Meta2 ', Text.meta2),
        ('Title ', Text.title),
        ('Heading ', Text.heading),
        ('Value ', Text.value),
        ('Highlight ', Text.highlight),
        ('Subtle', Text.subtle)
    ])


.. raw:: html

    <pre><span style="text-decoration: underline">Styles </span><span style="color: #E75C58">Danger </span><span style="color: #DDB62B">Warning </span><span style="color: #208FFB">Meta </span><span style="color: #60C6C8">Key </span><span style="color: #D160C4">Meta2 </span><strong><span style="text-decoration: underline">Title </span></strong><span style="text-decoration: underline">Heading </span><strong>Value </strong><strong><span style="color: #DDB62B">Highlight </span></strong><span style="color: #C5C1B4">Subtle</span></pre>


Or, specify colors


.. code-block:: python

    logger.log([
        ('Colors ', Text.heading),
        ('Red ', Color.red),
        ('Black ', Color.black),
        ('Blue ', Color.blue),
        ('Cyan ', Color.cyan),
        ('Green ', Color.green),
        ('Orange ', Color.orange),
        ('Purple Heading ', [Color.purple, Text.heading]),
        ('White', Color.white),
    ])

.. raw:: html

    <pre><span style="text-decoration: underline">Colors </span><span style="color: #E75C58">Red </span><span style="color: #3E424D">Black </span><span style="color: #208FFB">Blue </span><span style="color: #60C6C8">Cyan </span><span style="color: #00A250">Green </span><span style="color: #DDB62B">Orange </span><span style="color: #D160C4"><span style="text-decoration: underline">Purple Heading </span></span><span style="color: #C5C1B4">White</span></pre>


Logging debug info
-------------------

You can pretty print python objects with :meth:`info`.

.. code-block:: python

    logger.info(a=2, b=1)

.. raw:: html

    <pre><span style="color: #60C6C8">a: </span><strong>2</strong>
    <span style="color: #60C6C8">b: </span><strong>1</strong></pre>



.. code-block:: python

    logger.info(dict(name='Name', price=22))

.. raw:: html

    <pre><span style="color: #60C6C8"> name: </span><strong>Name</strong>
    <span style="color: #60C6C8">price: </span><strong>22</strong>
    Total <span style="color: #208FFB">2</span> item(s)</pre>
