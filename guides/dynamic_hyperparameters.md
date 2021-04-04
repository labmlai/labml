*Hyper-parameter values or schedules in model training are set before training starts either by intuition or through a hyper-parameter search. They either stay static or change based on a pre-determined schedule. We are introducing dynamic hyper-parameters which can be manually adjusted during the training. Researchers can adjust dynamic hyper-parameters based on model training stats.*

[Screenshot]

### What are hyper-parameters?

Hyper-parameters are parameters that control the learning process of models, such as the learning rate and weight decay. The model might not learn if you the hyper-parameters are not set right. Setting these is a key part of deep learning research. Researchers find them based on intuition or by running a hyper-parameter search. Hyper-parameter search is training a model repeatedly with different hyper-parameters to find the best set of hyper-parameters. Although, there are many techniques that speed up the hyper-parameter search, it's not feasible when there are many hyper-parameters and training the model takes a long time.

Some hyper-parameters are set based on a schedule. That is, the value of the hyper-parameter is changed based on a pre-determined schedule as the model trains. For example, you could slowly decrease the learning rate with time, or you could decrease the coefficient of an auxiliary loss as the model learns. Finding these schedules is nearly impossible with a hyper-parameter search, and are usually determined based on intuition of the researchers.

### Why is it hard to determine hyper-parameters?

Setting hyper-parameters require quite a bit of experience with the kind of models and sizes you are training and the dataset or the problem. Let take two examples: fine-tuning a language model and training a reinforcement learning agent. When you are fine-tuning a pre-trained langauge model, say to classify tweets, you get a pre-trained langauge model as the backbone and attach a single layer (may be two) classification head to it. First you freeze the parameters of the backbone and train the head for certain number of updates, and then you unfreeze all parameters and train all the parameters. The number of steps to keep the backbone frozen is generally set to 1 epoch. This is a hyper-parameter. And the common practice of freezing for 1 epoch might not suit your dataset. You would want to try a few numbers and look at learning curves and determine what works best.

Now consider setting the reward discount factor in an reinforcement learning agent. This determine how much the future rewards are discounted when considering the current step. A lower discount factor will only give rewards from next few steps, whilst a discount factor close to one will get rewards from all future steps. That is a smaller discount factor will make the agent short sighted. It's generally faster to train agents initially with a small discount factor and increase it to be close to one towards the end of the training. Knowing how fast to change this is difficult. You will know by intruition if you have trained agents on the same environment before. Otherwise you will have to run a few training sessions and get a better understanding after observing the training statistics.

## Introducing Dynamic Hyper-parameters

Dynamic hyper-parameters are hyper-parameters that researchers can adjust while the model is being trained. This allows researchers to actively control how the model trains, instead of letting the model train with a set of pre-determined set of hyper-parameters.

### Example use-cases

**Freezing pre-trained layers**: When fine-tuning a langauge model, you can train with the backbone frozen until the rate of improvement of loss drops, and change the hyper-parameter affecting which layers are frozen. This is better than going with the common practice of keeping the backbone frozen for 1 epoch for any dataset.

**Learning-rate warmup and decay**: Learning rate can be manually increase during the initial training updates. You could decide how long to warmup for based on the loss curves. Similarly you can decay the learning rate when the loss values appear to stabalize. This allows you to use higher learning rates initially to speed up the training.

**Increase sequence length**: Recurrent models train faster when the BPTT length is shorter. But you need higher BPTT lengths for better performance. Therefore, it is a common practice to start with a shorter BPTT length and increase it later. Again deciding when to do this before-hand is hard. Changign this dynamically is a lot easier.

**Adjusting regularization parameters**: You can also start with lower weight decay and lower dropout probabilities initially. Espeically if you are not sure about the represeentation capacity of the model. You can then increase these regualarization parameters later when the validation loss stops improving (higher variance).

**Adjusting reinforcement learning hyper-parameters**: Reinforment learning tend to have more hyper-parameters. Especially those that need to change during training, such as discount factor, entropy-bonus coefficients, learning-rate, etc. Changing these during training based on agents performance and other stats is a lot easier. Pre-determining them is almost impossible without observing a few training runs, and those training runs go many hours or days even for simple gaming environments.

### How it works?

You need to create a dynamic hyper parameter and register them along with other configurations.

```python
from labml import experiment
from labml.configs import FloatDynamicHyperParam

lr = FloatDynamicHyperParam(2.5e-4, range_=(0, 0.1))

experiment.configs({
  'learning_rate': lr,
  ...,
})
```



Next you just call the dynamic hyper-parameter to get the current value. For example:

```python
def train(batch):
  optimizer.set_lr(lr())
  optimizer.step()
```

The call `lr()` will return the current learning-rate set in [labml.ai](https://labml.ai) app.

[screenshot]

Here's a screen the mobile web interface for changing dynamic hyper-parameters. In this demo we adjusted learning-rate, clipping range, and number of training epochs (per sample) to speed up training of an [PPO agent](https://nn.labml.ai/rl/ppo/experiment.html). A standard weight decay and other static hyper-parameter values would cause it to take a lot of training updates to get over the score of 1.

### What's next

**Updating hyper-parameter schedules**: Our current implementation only allows users to update hyper-parameter values. This can sometimes take too much of user time. For instance, lets say based on current loss curves the user figures out that he wants to drop the learning rate from `1e-3` to `2.5e-4`, but instead of changing it now, he wants to do it slowly during the next 100,000 updates. With our current implementation he would have to manually change it little-by-little several times. We want to let users set and update hyper-parameter schedules, so that user has to manually intervene only when necessary.

**Rewind**: Often when training with dynamic hyper-parameters you feel like experimenting with them. Sort of like a small hyper-parameter search while the model is training. But when things go wrong it's hard to get things back on track. To overcome this we are working on a simple rewind (or undo) option, where the user could restart at an old checkpoint. This is possible even in current setup, but being able to do that with a couple of taps on the screen will make a big difference.