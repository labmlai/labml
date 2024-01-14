# Set Up

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

Please be aware that the above command utilizes port 5005 as the default. If you wish to use a different port, use the following command:

```commandline
 labml app-server --port PORT
```

Please access the UI by visiting `http://localhost:{port}` or, if set up on a different machine, use `http://{server-ip}:{port}`.
