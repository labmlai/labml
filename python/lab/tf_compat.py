import tensorflow as tf

TF_VERSION = [int(s) for s in tf.__version__.split('.')]
TF_VERSION = TF_VERSION[0] * 1000 + TF_VERSION[1]

if TF_VERSION < 1014:
    Session = tf.Session
    global_variables_initializer = tf.global_variables_initializer
    set_random_seed = tf.set_random_seed
    summary = tf.summary
    make_tensor_proto = tf.make_tensor_proto
    Summary = tf.Summary
    HistogramProto = tf.HistogramProto
else:
    Session = tf.compat.v1.Session
    global_variables_initializer = tf.compat.v1.global_variables_initializer
    set_random_seed = tf.compat.v1.set_random_seed
    summary = tf.compat.v1.summary
    make_tensor_proto = tf.compat.v1.make_tensor_proto
    Summary = tf.compat.v1.Summary
    HistogramProto = tf.compat.v1.HistogramProto
