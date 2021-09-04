[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai)

# Machine Learning Best Practices

## Coding

It's easy to make mistakes that are very hard to catch in ML. And debugging is hard too. So avoiding bugs in the first place helps a lot.

### Use version controlling

Use a version controlling system like **git** and commit your code after each minor change with good commit messages.

### Good coding style

Write beautiful, understandable, and clean code. Clean code is easy to read so it's easy to spot errors and therefore less likely to make errors.

Stick to practices such as **declaring constants**, **meaningful names**, **type hints**, and **named parameters**. They help when the code gets complex.

Sometimes you have quickly to test stuff. In those cases, it makes sense to write crappy code as long as you can clean it up before you commit it.

### Good software design

Use standard software design principles. Try to make the code modular and have the modules decoupled from each other.

### Comment the code

Comment complex logic and equations to improve code readability. It also helps write bug-free code.

### Proofread the code

Read your code after writing it. You'd be suprised how well this works for catching mistakes.

### Start with the simplest, and add one by one

Code the simplest version first. Test it and then add the complexities one by one. So if something degrades the performance or doesn't improve as much as you want it to, you know what's causing the it.

### Test

Test each component; at least the complex ones which are most likely to be buggy. You can write unit tests if that's possible. Small scripts that visualize the outputs of functions can act as manual tests. These tests will also act as documented examples.

### Measure execution time

Most machine learning code runs for hours and days. So speed is important. Therefore, it's important to know what's taking time and you should measure execution times for different components and tasks.

### Use an IDE

Features like **refactoring**, **usage search**, and **go to definition** are quite useful when writing code as well as when reading. Most IDE's also have auto-complete and error highlighting that help you code faster with less errors.

### Use compiled code for slow operations

Sometimes it's hard to vectorize operations to make them run faster. You can write **cpython** modules or use just-in-time compilers like **numba**, in such cases to speed things up.

### Write the core of the algorithm yourself

When it's your code, it's easier to try variations. If you use libraries or unfamiliar code for core components, it will be difficult later when you want to make changes.

## Data

In real-world AI, data processing and preparation take most of your time compared to model building.

### Process data in steps

Process data one step at a time. This will make your data processing code modular and you will be able to reuse these components across different projects.

### Cache processed data

Cache data after each processing step. This will save you a lot of computation time and you will be able to check and analyze data at every stage. Storage is a lot cheaper compared to compute. Use computer-friendly formats like numpy arrays, instead of csv or json to load faster.

## Notebooks

Notebooks are popular among data scientists. They are good for a lot of things, but not so much when it comes to maintainability and reusability. They don't work well with version controlling and they can't be imported for reuse.

### Notebooks are good for trying things

Notebooks help you try stuff quickly. You can try things step by step in cells while checking the results at each stage. You can rewrite them in Python after trying them out on notebooks.

### Notebooks are great for visualizations

You can do custom analyses and visualizations on notebooks. You can use it for analyzing models, data, and training statistics.

### Notebooks are great for data loading and parsing

On notebooks, you can check and test each step as you are parsing data. You can output samples and statistics after each processing step. Since the state is in memory, you don't have to re-run all the previous steps when you make a change.

## Experiments

It's important to keep track of experiments when doing research. So you know what worked and what didn't. It is useful when you want to figure out what will work.

### Keep experiment statistics

Keep track of the results of all your experiments. Keep the performance statistics of the experiments with reference to the code and datasets.

### Save models

You should save the trained models because there's a good chance that you will need to analyze them later.

### Log more details

It's important to log details instead of summarized stats. For example, you can get a lot more insights by looking at a histogram of losses instead of the mean.

## Production

### Save inputs and outs

You will need them to train new models as well as to compare new models with the existing model. If saving them is not possible, consider saving a sample of them. You can sample randomly or based on model uncertainty.

### Analyze input and output distributions

When possible watch for changes in the distributions. For instance, if you had a cat/dog classifier you should monitor the cat to dog ratio over time. Changes to this can indicate problems.

### Monitor efficiency

Monitor how much time it takes to make predictions and the throughput. This will tell you whether you need to improve efficiencies.

### Run new models in parallel

Before deploying new models try to run them in parallel with existing ones and analyze where they differ.

### Compare with undeployed models

If you have deployed faster and smaller models (distilled), compare their inferences with the undeployed larger models.

## General

### Figure out evaluation criteria for the task

Make sure you have proper evaluation criteria. This is different from the loss function. It should be directly related to the final outcome of the project. For example, if the final benefit of your model is saving money, your final evaluation must be based on the amount of money saved.

This will help you make decisions on how much more time to invest in improving the models. It will also be necessary when you want to compare models with different loss functions.

### Maintain a journal
Journals help organize ideas. It will act as a reference and will help you avoid repeating mistakes. It will also help you get back on track when you come back to a project after a break.

### Track your time
Keep track of how much time you are spending on different tasks. It's easy to waste time on dead ends.

### Stay up to date

The machine learning field is advancing quite fast, so it's easy to get outdated. Join reading groups and allocate time to read papers. Spend time trying new tools and libraries. Conference websites, Twitter and Reddit help you find the latest tools and research.

## Resources

* [How to avoid machine learning pitfalls: a guide for academic researchers](https://papers.labml.ai/paper/2108.02497) by [Michael A. Lones](https://twitter.com/michael_lones)
* [Pitfalls in Machine Learning Research: Reexamining the Development Cycle](https://papers.labml.ai/paper/2011.02832) by [Stella Biderman](https://twitter.com/BlancheMinerva) and Walter J. Scheirer
* [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) by [Andrej Karpathy](https://twitter.com/karpathy)
* [21 Habits I Picked Up While Learning Machine Learning](https://blog.varunajayasiri.com/practices_learned_while_learning_machine_learning.html) by [Varuna Jayasiri](https://twitter.com/vpj)

Please feel free to contribute to this document.