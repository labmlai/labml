import os

import numpy as np
import tensorflow as tf


def init_variables(session: tf.Session):
    """
    #### Initialize TensorFlow variables
    """
    init_op = tf.global_variables_initializer()
    session.run(init_op)


def get_configs():
    """
    #### Default TF Configs
    """

    # let TensorFlow decide where to run operations;
    #  I think it chooses the GPU for everything if you have one
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)

    # grow GPU memory as needed
    config.gpu_options.allow_growth = True

    return config


def set_random_seeds(seed: int = 7):
    """
    #### Set random seeds
    """

    # set random seeds,
    np.random.seed(seed)
    tf.set_random_seed(seed)


def use_gpu(gpu: str):
    # I was using a computer with two GPUs and I wanted TensorFlow to use only one of them.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def strip_variable_name(name: str):
    if len(name) < 2:
        return name
    elif name[-2:] == ":0":
        return name[:-2]
    else:
        return name


def variable_name_to_file_name(name: str):
    assert name.find(":") == -1

    return name.replace("/", "__")