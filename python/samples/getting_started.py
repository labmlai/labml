"""
```trial
2019-07-28 16:10:06
Sample lab experiment
[[dirty]]: 🗑  remove guards, use https://github.com/vpj/guards
start_step: 0

--------------------------
| global_step |   reward |
--------------------------
|          10 |     1.50 |
|          20 |     4.83 |
|          30 |     8.17 |
|          40 |    11.50 |
--------------------------
```
"""

import lab.clear_warnings

import time

import tensorflow as tf

from lab import logger
from lab.experiment.tensorflow import Experiment


# Create the sample experiment
EXPERIMENT = Experiment(name="sample",
                        python_file=__file__,
                        comment="Sample lab experiment",
                        check_repo_dirty=False)

# Sections are use to keep track of
# what's going on from the console output.
# It is also useful to organize the code into sections,
# when separating them into functions is difficult
with logger.section("Create model"):
    # Indicate that this section failed. You don't have to set
    #  this if it is successful.
    logger.set_successful(False)

    # Sleep for a minute.
    time.sleep(1)

# Print sample info
logger.info(one=1,
            two=2,
            string="string")

# ### Set logger indicators

# Reward is queued; this is useful when you want to track the moving
# average of something.
logger.add_indicator("reward", queue_limit=10)

# By default everything is a set of values and will create a TensorBoard histogram
# We specify that `fps` is a scalar.
# If you store multiple values for this it will output the mean.
logger.add_indicator("fps", is_histogram=False, is_print=False)

# This will produce a histogram
logger.add_indicator("loss", is_print=False)

# A heat map
logger.add_indicator("advantage_reward", is_histogram=False, is_print=False, is_pair=True)

# Create a TensorFlow session
with tf.Session() as session:
    # Start the experiment from scratch, without loading from a
    # saved checkpoint ('is_init=True')
    # This will clear all the old checkpoints and summaries for this
    # experiment.
    # If you start with 'is_init=False',
    # the experiment will load from the last saved checkpoint.
    EXPERIMENT.start_train(session, is_init=True)

    # This is the main training loop of this project.
    for global_step in logger.loop(range(50)):
        # You can set the global step explicitly with
        # 'logger.set_global_step(global_step)'

        # Handle Keyboard Interrupts
        try:
            with logger.delayed_keyboard_interrupt():
                # A sample monitored section inside iterator
                with logger.section("sample"):
                    time.sleep(0.5)

                # A silent section is used only to organize code.
                # It produces no output
                with logger.section("logging", is_silent=True):
                    # Store values
                    logger.store(
                        reward=global_step / 3.0,
                        fps=12
                    )
                    # Store more values
                    for i in range(global_step, global_step + 10):
                        logger.store('loss', i)
                        logger.store(advantage_reward=(i, i * 2))

                # Another silent section
                with logger.section("process_samples", is_silent=True):
                    time.sleep(0.5)

                # A third section with an inner loop
                with logger.section("train", total_steps=100):
                    # Let it run for multiple iterations.
                    # We'll track the progress of that too
                    for i in range(100):
                        time.sleep(0.01)
                        # Progress is tracked manually unlike in the top level iterator.
                        # The progress updates do not have to be sequential.
                        logger.progress(i + 1)

                # Log stored values.
                # This will output to the console and write TensorBoard summaries.
                logger.write()

                # Store progress in the trials file and in the python code as a comment
                if (global_step + 1) % 10 == 0:
                    logger.save_progress()

                # By default we will overwrite the same console line.
                # `new_line` makes it go to the next line.
                # This helps keep the console output concise.
                if (global_step + 1) % 10 == 0:
                    logger.new_line()
        except KeyboardInterrupt:
            logger.finish_loop()
            logger.new_line()
            logger.log(f"Stopping the training at {global_step} and saving checkpoints")
            break

with logger.section("Cleaning up"):
    time.sleep(0.5)
