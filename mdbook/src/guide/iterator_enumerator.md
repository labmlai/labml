### Iterator and Enumerator

You can use `logger.iterate` and `logger.enumerate` with any iterable object.
In this example we use a PyTorch `DataLoader`.


```python
# Create a data loader for illustration
import torch
from torchvision import datasets, transforms

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=False,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=32, shuffle=True)
```


```python
for data, target in logger.iterate("Test", test_loader):
    time.sleep(0.01)
```


<pre>Test<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	6,618.08ms</span>
</pre>



```python
for i, (data, target) in logger.enum("Test", test_loader):
    time.sleep(0.01)
```


<pre>Test<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	6,662.78ms</span>
</pre>
