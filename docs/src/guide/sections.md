
### Sections

Sections let you monitor time taken for
different tasks and also helps *keep the code clean*
by separating different blocks of code.


```python
import time
```


```python
with logger.section("Load data"):
    # code to load data
    time.sleep(2)
```


<pre>Load data<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	2,005.04ms</span>
</pre>



```python
with logger.section("Load saved model"):
    time.sleep(1)
    logger.set_successful(False)
```


<pre>Load saved model<span style="color: #E75C58">...[FAIL]</span><span style="color: #208FFB">	1,008.86ms</span>
</pre>


You can also show progress while a section is running


```python
with logger.section("Train", total_steps=100):
    for i in range(100):
        time.sleep(0.1)
        # Multiple training steps in the inner loop
        logger.progress(i)
```


<pre>Train<span style="color: #00A250">...[DONE]</span><span style="color: #208FFB">	10,600.04ms</span>
</pre>