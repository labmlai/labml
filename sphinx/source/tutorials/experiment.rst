Experiment
==========

As the final step, you need to start and run the experiment. Lab provides a convenient way to do this.


.. code-block:: python

    def main():
        conf = Configs()
        experiment = Experiment(writers={'sqlite', 'tensorboard'})
        experiment.calc_configs(conf,
                                {'optimizer': 'adam_optimizer'},
                                ['set_seed', 'run'])
        experiment.add_models(dict(model=conf.model))
        experiment.start()
        conf.main()


    if __name__ == '__main__':
        main()

Note that in the above code snippet, We have declared an Experiment and passed the writers, in this case,  `sqlite` and `tensorboard`. By default Lab will writes every log to the console. Moreover, you can pass the order of calculating configs by passing a list of the order in experiment.calc_configs.

