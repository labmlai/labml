### Experiment

Lab will keep track of experiments if you declare an Experiment. It will keep track of logs, code diffs, git commits, etc.


```python
from lab.experiment.pytorch import Experiment
```

The `name` of the defaults to the calling python filename. However when invoking from a Jupyter Notebook it must be provided because the library cannot find the calling file name. `comment` can be changed later from the [Dashboard](https://github.com/vpj/lab_dashboard).


```python
exp = Experiment(name="mnist_pytorch",
                 comment="Test")
```

Starting an experiments creates folders, stores the experiment meta data, git commits, and source diffs.


```python
exp.start()
```


<pre><strong><span style="text-decoration: underline">mnist_pytorch</span></strong>: <span style="color: #208FFB">a4a7586664f411eab095acde48001122</span>
	<strong><span style="color: #DDB62B">Test</span></strong>
	[dirty]: <strong><span style="color: #DDB62B">"Update readme.md"</span></strong></pre>


You can also start from a previously saved checkpoint. A `run_uuid` of `` means that it will load from the last run.

```python
exp.start(run_uuid='')
```
