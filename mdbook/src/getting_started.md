## Getting Started

### Clone and install

```bash
git clone git@github.com:vpj/lab.git
cd lab
pip install -e .
```

To update run a git update

```bash
cd lab
git pull
```

<!-- ### Install it via `pip` directly from github.

```bash
pip install -e git+git@github.com:vpj/lab.git#egg=lab
``` -->

### Create a `.lab.yaml` file.
An empty file at the root of the project should
be enough. You can set project level configs for
 'check_repo_dirty' and 'path'
in the config file.

Lab will store all experiment data in folder `logs/` 
relative to `.lab.yaml` file.
If `path` is set in `.lab.yaml` then it will be stored in `[path]logs/`
 relative to `.lab.yaml` file.

You don't need the `.lab.yaml` file if you only plan on using the logger.

## Lab Dashboard

### Clone and install

```bash
git clone git@github.com:vpj/lab_dashboard.git
cd lab_dashboard
git submodule init
git submodule update
./install.sh
```

To update run a git update

```bash
cd lab_dashboard
git pull
git submodule update
./install.sh
```

### Starting the server

Navigate to the path of the project and run the following command to start the server.

```bash
lab_dashboard
```