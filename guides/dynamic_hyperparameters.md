*Hyper-parameters control the learning process of models. They are set before training starts, either by intuition or a hyper-parameter search. They either stay static or change based on a pre-determined schedule. We are introducing dynamic hyper-parameters which can be manually adjusted during the training based on model training stats.*

<div align="center">
  <img src="https://github.com/lab-ml/labml/raw/master/guides/dynamic_hp_desktop.png" width="600px" alt="Dynamic Hyper-parameters Mobile Screenshot"/>
</div>

### What are hyper-parameters?

Hyper-parameters are parameters that control the learning process of models, such as the learning rate, batch size, and weight decay. The model might not learn if the hyper-parameters are not set correctly. Setting hyper-parameters is a key part of deep learning research. Researchers find them based on intuition or by running a hyper-parameter search. In a hyper-parameter search, the model is trained repeatedly with different hyper-parameters to find the best set of hyper-parameters. Hyper-parameter search becomes costly as the number of hyper-parameters increases and the model training time increases.

Some hyper-parameters change during the training based on a pre-determined schedule. For example, you could slowly decrease the learning rate, or you could decrease the coefficient of an auxiliary loss as the model learns. Finding these schedules is nearly impossible with a hyper-parameter search, and are usually determined based on the intuition of the researchers.

### Why is it hard to determine hyper-parameters?

Setting hyper-parameters require quite a bit of experience with the kind of models and sizes you are training as well as the dataset. For instance, consider fine-tuning a pre-trained language model to classify tweets. You get a pre-trained language model as the backbone and attach a layer (or two) as the classification head. First, you freeze the parameters of the backbone and train the head for a certain number of updates, and then you unfreeze all parameters and train all the parameters. The number of steps to keep the backbone frozen is generally set to 1 epoch. This is a hyper-parameter. And the common practice of freezing for 1 epoch might be too small or too large depending on the size of the model as well as the dataset size. Someone who has worked with similar models and datasets will have a good intuition on this hyper-parameter. If you are new, you will have to try training the model to get a feel about it.

<!-- Now consider setting the reward discount factor in a reinforcement learning agent. This determines how much the future rewards are discounted when considering the current step. A lower discount factor will only give rewards from the next few steps, whilst a discount factor close to one will get rewards from all future steps. That is, a smaller discount factor will make the agent short-sighted. It's generally faster to train agents initially with a small discount factor and increase it to be close to one towards the end of the training. Knowing how fast to change this is difficult. You will know by intuition if you have trained agents in the same environment before. Otherwise, you will have to run a few training sessions and study them to get a better understanding. -->

## ⚙️ Introducing Dynamic Hyper-parameters

Dynamic hyper-parameters are hyper-parameters that researchers can adjust while the model is being trained. This allows researchers to actively control how the model trains, instead of letting the model train with a pre-determined set of hyper-parameters. Dynamic hyper-parameters help train the model faster and better on a single training session. Also, they let researchers play around with the hyper-parameters during a single training run to gather insights.

Sometimes researchers save the model checkpoints and restart the training with changed hyper-parameter values. This has a similar effect to dynamic hyper-parameters but it's quite cumbersome to do.

### How does it work?

You need to create a dynamic hyper-parameter and register them along with other configurations.

```python
from labml import experiment
from labml.configs import FloatDynamicHyperParam

lr = FloatDynamicHyperParam(2.5e-4, range_=(0, 0.1))

experiment.configs({
  'learning_rate': lr,
  ...,
})
```

Then can call the dynamic hyper-parameter to get the current value. For example:

```python
def train(batch):
  optimizer.set_lr(lr())
  optimizer.step()
```

The call `lr()` will return the current learning rate set in [labml.ai](https://labml.ai) [app](https://github.com/lab-ml/app).

<div align="center">
  <img src="https://github.com/lab-ml/labml/raw/master/guides/dynamic_hp.png" width="400px" alt="Dynamic Hyper-parameters Mobile Screenshot"/>
</div>

*Above is a screenshot of the mobile web interface for changing dynamic hyper-parameters. In this [![Demo](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/6eff28a0910e11eb9b008db315936e2f/hyper_params) demo we adjusted the learning rate, clipping range, and the number of training epochs (per sample) to speed up the training of a [PPO agent](https://nn.labml.ai/rl/ppo/experiment.html) for Atari Breakout. A standard learning rate decay and other static hyper-parameter values would have taken a lot of training updates to get over the score of 1.*

### Example use-cases

**Freezing pre-trained layers**: When fine-tuning a language model, you can train with the backbone frozen until the rate of improvement of loss drops, and change the hyper-parameter affecting which layers are frozen. This is better and faster than going with the common practice of keeping the backbone frozen for 1 epoch for all models and datasets.

**Learning-rate warm-up and decay**: The learning rate can be manually increased during the initial training updates. You could decide how long to warm up for based on the loss curves. Similarly, you can decay the learning rate when the loss values stabilize. This allows you to use higher learning rates initially to speed up the training.

**Increase sequence length**: Recurrent models train faster when the BPTT length is shorter. But you need higher BPTT lengths to improve accuracy. Therefore, it is a common practice to start with a shorter BPTT length and increase it later. Again deciding when to do this beforehand is hard. Changing this dynamically is a lot easier.

**Adjusting regularization parameters**: You can start with lower weight decay and lower dropout probabilities initially. Especially if you are not sure about the representation capacity of the model. You can then increase these regularization parameters later when the validation loss stops improving (higher variance).

**Adjusting reinforcement learning hyper-parameters**: Reinforcement learning tends to have more hyper-parameters. Most of which need to change during training, such as discount factor, entropy-bonus coefficients, learning-rate, etc.  Pre-determining them is almost impossible without observing a few training runs, and those training runs go many hours or days even for simple gaming environments. Changing these during training based on the agent's performance and other stats is a lot easier.

### What's next

**Updating hyper-parameter schedules**: Our current implementation only allows users to update hyper-parameter values. This can take too much user time. For instance, let's say based on current loss curves the user figures out that he wants to drop the learning rate from `1e-3` to `2.5e-4` during the next 100,000 updates. With our current implementation, she would have to make several manual  changes. We want to let users set and update hyper-parameter schedules so that user has to manually intervene only when necessary.

**Rewind**: Often when training with dynamic hyper-parameters you feel like experimenting with them. Sort of like a small hyper-parameter search while the model is training. But when things go wrong you want to reset. To enable this we are working on a simple rewind (or undo) option, where the user could restart at  any checkpoint, with a couple of taps on the screen.