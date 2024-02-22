# Hosting Your Own Experiments Server

## 1. Server set up

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

### Installing the Packages

```commandline
pip install labml labml-app -U
```

### Starting the Server

```commandline
 labml app-server
```

Please be aware that the above command utilizes port `5005` as the default. If you wish to use a different port, use the
following command:

```commandline
 labml app-server --port PORT
```

Please access the UI by visiting `http://localhost:{port}` or, if set up on a different machine,
use `http://{server-ip}:{port}`.

## 2. Monitor experiments

Create a file named `.labml.yaml` at the top level of your project folder.

Add the following line to the file:

```yaml
app_url: http://localhost:{port}/api/v1/default
```

If you are setting up the project on a different machine, include the following line instead:

```yaml
app_url: http://{server-ip}:{port}/api/v1/default
```

Please follow the examples given in Readme.md 

## 3. Nginx setup (optional)

### Install Nginx

```commandline
sudo apt install nginx
```

### Start Nginx

```commandline
systemctl start nginx
```

### Configuration

Generate a file called `labml_app.conf` within the `/etc/nginx/sites-available` directory, and include the following
content.

```nginx configuration
server {
        listen 80;
        listen <port>;
        server_name <server-ip>;

        location / {
               proxy_pass  http://127.0.0.1:<port>;
               proxy_set_header Host $http_host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

}
```

Enable the file by creating a link to it within the `sites-enabled` directory.

```commandline
sudo ln -s /etc/nginx/sites-available/labml_app.conf /etc/nginx/sites-enabled/
```

Restart the Nginx service.

```commandline
sudo systemctl restart nginx
```

You can
follow [this guide](https://www.digitalocean.com/community/tutorials/how-to-improve-website-performance-using-gzip-and-nginx-on-ubuntu-20-04)
to configure `Nginx` to use `Gzip` for data compression.