Experiment
==========

.. automodule:: labml.experiment

     Create Experiment
     -----------------

     .. autofunction:: create

     .. autofunction:: record

     .. autofunction:: evaluate

     .. autofunction:: start

     .. autofunction:: get_uuid


     Checkpoints
     -----------

     .. autofunction:: load

     .. autofunction:: load_models

     .. autofunction:: save_checkpoint

     .. autofunction:: add_pytorch_models

     .. autofunction:: add_sklearn_models

     .. autofunction:: add_model_savers

     .. autoclass:: ModelSaver

          .. automethod:: save

          .. automethod:: load


     Configurations & Hyper-parameters
     ---------------------------------

     .. autofunction:: configs

     .. autofunction:: load_configs


     Bundle checkpoints
     ------------------

     .. autofunction:: save_bundle

     .. autofunction:: load_bundle


     Distirbuted training
     --------------------

     .. autofunction:: distributed


     Utilities
     ---------

     .. autofunction:: save_numpy
