Helpers
=======

**Installation**

.. code-block:: console

    pip install labml_helpers

Configurable Modules
--------------------

.. autoclass:: labml_helpers.device.DeviceConfigs

.. autoclass:: labml_helpers.seed.SeedConfigs

.. autoclass:: labml_helpers.optimizer.OptimizerConfigs

.. autoclass:: labml_helpers.training_loop.TrainingLoopConfigs

.. autoclass:: labml_helpers.train_valid.TrainValidConfigs

.. autoclass:: labml_helpers.train_valid.SimpleTrainValidConfigs

Datasets
--------

.. autoclass:: labml_helpers.datasets.mnist.MNISTConfigs

.. autoclass:: labml_helpers.datasets.cifar10.CIFAR10Configs

.. autoclass:: labml_helpers.datasets.csv.CsvDataset

Text Datasets
^^^^^^^^^^^^^

.. autoclass:: labml_helpers.datasets.text.TextDataset

.. autoclass:: labml_helpers.datasets.text.TextFileDataset

.. autoclass:: labml_helpers.datasets.text.SequentialDataLoader

.. autoclass:: labml_helpers.datasets.text.SequentialUnBatchedDataset

Schedules
---------

.. autoclass:: labml_helpers.schedule.Schedule

.. autoclass:: labml_helpers.schedule.Flat

.. autoclass:: labml_helpers.schedule.Dynamic

.. autoclass:: labml_helpers.schedule.Piecewise

.. autoclass:: labml_helpers.schedule.RelativePiecewise


Metrics
-------

.. autoclass:: labml_helpers.metrics.StateModule

.. autoclass:: labml_helpers.metrics.Metric

.. autoclass:: labml_helpers.metrics.accuracy.Accuracy

.. autoclass:: labml_helpers.metrics.accuracy.BinaryAccuracy

.. autoclass:: labml_helpers.metrics.accuracy.AccuracyDirect

.. autoclass:: labml_helpers.metrics.collector.Collector

.. autoclass:: labml_helpers.metrics.recall_precision.RecallPrecision

.. autoclass:: labml_helpers.metrics.simple_state.SimpleStateModule

.. autoclass:: labml_helpers.metrics.simple_state.SimpleState

Utilies
---------

.. autoclass:: labml_helpers.module.Module
