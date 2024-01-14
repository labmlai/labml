# 1. Set Up

### Installing MongoDB

To install MongoDB, refer to the official
documentation [here](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/).

Initiate MongoDB Service:

```commandline
sudo systemctl start mongod
```

Verify that MongoDB is running with the following command:

```commandline
sudo systemctl status mongod
```

### Installing the Package

```commandline
pip install labml-app -U
```

### Starting the Server

```commandline
 labml app-server
```

Please be aware that the above command utilizes port 5005 as the default. If you wish to use a different port, use the
following command:

```commandline
 labml app-server --port PORT
```

Please access the UI by visiting http://localhost:{port}/ or, if set up on a different machine, use http://{server-ip}:
{port}.

# 2. Experiment Monitoring

Create a file named `.labml.yaml` at the top level of your project folder.

Add the following line to the .labml.yaml file for local setups:

```yaml
app_url: http://localhost:{port}/api/v1/default
```

If the setup is on a different machine, include the following line instead:

```yaml
app_url: http://{server-ip}:{port}/api/v1/default
```

Ensure to replace `{server-ip}` and `{port}` with the appropriate values for your specific configuration.

# 3. Hardware Monitoring

### Installation of Required Packages and Dependencies

```commandline
pip install labml psutil py3nvml
```

### Initiating Monitoring

```commandline
labml monitor
```




