[![PyPI - Python Version](https://badge.fury.io/py/labml.svg)](https://badge.fury.io/py/labml)
[![Twitter](https://img.shields.io/twitter/follow/labmlai?style=social)](https://twitter.com/labmlai?ref_src=twsrc%5Etfw)

# Hardware monitoring

<div align="center">
<img src="https://github.com/labmlai/labml/blob/master/guides/hardware.png" width="30%" alt=""/>
</div>

[labml.ai](https://labml.ai) app can be used to monitor hardware.

*You can use it to monitor hardware usage even if you are not training models.*

### Installation

You need to install our python package and some dependencies to start monitoring.

```sh
pip install labml psutil py3nvml
```

* `psutil` is the package we use to track hardware (`labml` doesn't install it as a dependency).
* To monitor GPUs (nvidia) you need to install `py3nvml` package as well.

If you have problems with your Python environment you can use
[our guide to setting up Python with conda locally](https://github.com/labmlai/labml/blob/master/guides/local-ubuntu.md),
or [on a remote computer](https://github.com/labmlai/labml/blob/master/guides/remote-python.md).

### Configs

You can set the url of the server in  `~/.labml/configs.yaml`.

```yaml
app_url: https://hosted-labml-app.com/api/v1/computer
```

*Note that the `&` at the end is important.*
### Start monitoring

Once it's installed you can just run the following command to start monitoring.
You will get a url to view the hardware usage.

```sh
labml monitor
```

You can also run it with `nohup` if you want it to run in the background on a remote computer.

```sh
nohup labml monitor &
```

### Setting up a service to monitor

You can setup a [`systemd`](https://systemd.io/) service to monitor your computer.
This will make sure your computer gets monitored in the background and the service should start upon computer restarts.

```sh
labml service
```

Above command will setup the service and show you how to start/stop the service.
