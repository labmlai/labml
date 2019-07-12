import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from lab import util


def random_string(length=10):
    import random
    import string

    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def save_labels(path, labels):
    with open(str(path), 'w') as f:
        for i, label in enumerate(labels):
            f.write(f"{label}\n")


def create_sprite_image(images: np.ndarray):
    if len(images.shape) == 3:
        images = images.reshape((*images.shape, 1))
    n_images, height, width, channels = images.shape

    rows = cols = int(math.ceil(math.sqrt(n_images)))

    sprite = np.ones((rows * height, cols * width, channels))

    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            if n >= n_images:
                continue
            image = images[n]
            sprite[r * height: (r + 1) * height, c * width: (c + 1) * width, :] = image

    return sprite


def save_sprite_image(path, images: np.ndarray):
    sprite = create_sprite_image(images)
    sprite *= 255.
    if sprite.shape[2] == 1:
        sprite = sprite.reshape((sprite.shape[:2]))
        sprite = scipy.misc.toimage(sprite)
    else:
        sprite = scipy.misc.toimage(sprite, channel_axis=2)

    sprite.save(str(path))


def save_embeddings(*,
                    path: Path,
                    embeddings: np.ndarray,
                    images: Optional[np.ndarray],
                    labels: Optional[List[str]]):
    if path.exists():
        util.rm_tree(path)
    path.mkdir(parents=True)

    sess = tf.compat.v1.InteractiveSession()
    name = random_string()

    embeddings_variable = tf.Variable(embeddings, trainable=False, name=name)

    tf.global_variables_initializer().run()

    saver = tf.compat.v1.train.Saver()
    writer = tf.compat.v1.summary.FileWriter(str(path), sess.graph)

    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embeddings_variable.name

    if labels is not None:
        embed.metadata_path = 'metadata.tsv'
        save_labels(path / 'metadata.tsv', labels)

    if images is not None:
        embed.sprite.image_path = "sprites.png"
        save_sprite_image(path / 'sprites.png', images)

        embed.sprite.single_image_dim.extend([images.shape[1], images.shape[2]])

    projector.visualize_embeddings(writer, config)

    saver.save(sess, str(path / 'a_model.ckpt'), global_step=0)


def my_main():
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('data/mnist_data',
                                      one_hot=False)
    xs, ys = mnist.train.next_batch(500)
    embeddings = xs
    images = 1 - embeddings.reshape(-1, 28, 28)
    labels = [str(label) for label in ys]

    save_embeddings(path=Path(os.getcwd()) / 'logs' / 'projector',
                    embeddings=embeddings,
                    images=images,
                    labels=labels)


if __name__ == '__main__':
    my_main()
