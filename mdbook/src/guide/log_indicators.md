### Log indicators

Here you specify indicators and the logger stores them temporarily and write in batches.
It can aggregate and write them as means or histograms.


```python
import numpy as np

# dummy train function
def train():
    return np.random.randint(100)

# Reset global step because we incremented in previous loop
logger.set_global_step(0)
```

This stores all the loss values and writes the logs the mean on every tenth iteration.
Console output line is replaced until `new_line` is called.


```python
for i in range(1, 401):
    logger.add_global_step()
    loss = train()
    logger.store(loss=loss)
    if i % 10 == 0:
        logger.write()
    if i % 100 == 0:
        logger.new_line()
    time.sleep(0.02)
```


<pre><strong><span style="color: #DDB62B">     100:  </span></strong> loss: <strong> 42.8000</strong>
<strong><span style="color: #DDB62B">     200:  </span></strong> loss: <strong> 39.0000</strong>
<strong><span style="color: #DDB62B">     300:  </span></strong> loss: <strong> 39.7000</strong>
<strong><span style="color: #DDB62B">     400:  </span></strong> loss: <strong> 53.6000</strong></pre>


#### Indicator settings


```python
from lab.logger.indicators import Queue, Scalar, Histogram

# dummy train function
def train2(idx):
    return idx, 10, np.random.randint(100)

# Reset global step because we incremented in previous loop
logger.set_global_step(0)
```

`Histogram` indicators will log a histogram of data.
`Queue` will store data in a `deque` of size `queue_size`, and log histograms.
Both of these will log the means too. And if `is_print` is `True` it will print the mean.

queue size of `10` and the values are printed to the console


```python
logger.add_indicator(Queue('reward', 10, True))
```

By default values are not printed to console; i.e. `is_print` defaults to `False`.


```python
logger.add_indicator(Scalar('policy'))
```

Settings `is_print` to `True` will print the mean value of histogram to console


```python
logger.add_indicator(Histogram('value', True))
```


```python
for i in range(1, 400):
    logger.add_global_step()
    reward, policy, value = train2(i)
    logger.store(reward=reward, policy=policy, value=value, loss=1.)
    if i % 10 == 0:
        logger.write()
    if i % 100 == 0:
        logger.new_line()
```


<pre><strong><span style="color: #DDB62B">     100:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 95.5000</strong> value: <strong> 40.5000</strong>
<strong><span style="color: #DDB62B">     200:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 195.500</strong> value: <strong> 53.4000</strong>
<strong><span style="color: #DDB62B">     300:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 295.500</strong> value: <strong> 60.3000</strong>
<strong><span style="color: #DDB62B">     390:  </span></strong> loss: <strong> 1.00000</strong> reward: <strong> 385.500</strong> value: <strong> 45.0000</strong></pre>

