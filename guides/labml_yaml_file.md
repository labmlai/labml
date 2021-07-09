# `.labml.yaml` configurations file

You should create a (possible empty) `.labml.yaml` file at the top of your project folder.

These configurations can be overridden with
[labml.lab.configure](https://docs.labml.ai/api/lab.html#labml.lab.configure)
function as well.

This file contains a set of configurations [labml](https://github.com/labmlai/labml)
uses. The configurations will default to values below if you do not specify them explicitly.

Configurations:

```yaml
    check_repo_dirty: true
    data_path: 'data'
    experiments_path: 'logs'
    analytics_path: 'analytics'
    web_api: 'TOKEN from web.lab-ml.com'
    web_api_frequency: 60
    web_api_verify_connection: true
    web_api_open_browser: true
    indicators:
      - class_name: Scalar
        is_print: True
        name: '*'
```

* `check_repo_dirty`: If `true`, before running an experiment it checks and aborts if there are any uncommitted changes

* `data_path`: The location of data files. This can be accessed via
  [`labml.lab.get_data_path`](https://docs.labml.ai/api/lab.html#labml.lab.get_data_path)

* `experiments_path`: This is where **labml** will store all the experiment details such as logs, configs and
  checkpoints. This can be accessed via
  [`labml.lab.get_experiments_path`](https://docs.labml.ai/api/lab.html#labml.lab.get_experiments_path)

* `analytics_path`:  This is where Jupyter Notebooks for custom analytics will be saved. ⚠️ This is still experimental.

* `web_api`: The token from [app.labml.ai](https://app.labml.ai) or an url to a self-hosted app.

* `web_api_frequency`: Interval in seconds to push stats to labml.ai app.

* `web_api_verify_connection`: Whether to verify SSL certificate of the app. You might want to set this to `false` if
  you self-host and use an unverified SSL certificate.

* `web_api_open_browser`: Whether to open the monitoring url in the browser automatically when the experiment starts.

* `indicators`: Use this to specify types of
  [indicators for tracker](https://colab.research.google.com/github/lab-ml/labml/blob/master/guides/tracker.ipynb).
    * `class_name` is the type of the indicator
    * `is_print` is whether to output the statistic to console.
    * `name` can be a wildcard selector for indicator names. You can set these individually with the
      [tracker API](https://colab.research.google.com/github/lab-ml/labml/blob/master/guides/tracker.ipynb).