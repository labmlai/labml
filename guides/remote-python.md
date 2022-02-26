# How to Setup a Remote Machine For Machine Learning

Setting up a remote machine to run a python project is a bit of an annoying task. Most of the new developers who are not much familiar, get stuck at some stage of the setting up process. This tutorial is intended to provide a good explanation about the setup process even for a complete newbie.

## 🖥 `SSH` to the remote machine

```sh
ssh -i "[PRIVATE_KEY_FILE_PATH]" [USERNAME]@[HOST_NAME]
```

**`[PRIVATE_KEY_FILE_PATH]`** Should be `FOLDER_PATH/sample.pem` or just `sample.pem` if the current directory is the same

**`[USERNAME]@[HOST_NAME]`** is something like `ubuntu@ec2-X-XXX-XXX-XXX.us-east-2.compute.amazonaws.com`where `ubuntu` is the username.

Rest of the commands should be run on the remote computer after `ssh`ing into it.

## Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

`python` may or may not be pre-installed in your remote machine. Even if it is pre-installed, you might need a different `python` version. Also, you might need different Python versions for different projects.

`Miniconda` solves the above problems for us. Not only its lightweight but also its allow us to create multiple `python`environments with different python versions as well as different python packages.

First, you need to download the Miniconda setup to your remote machine.

```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
```

Then, install it.

```sh
bash miniconda.sh -b -p "[CONDA_PATH]"
```

For example,

```sh
bash miniconda.sh -b -p ~/miniconda
```

**`[CONDA_PATH]`** is the path for the conda installation. We use `~/miniconda`. 

Even though we installed miniconda, if you run

```sh
conda -V
```

in your terminal, you will get an error `conda: command not found`.

To activate `conda`, you will need to run the following commands and restart your `ssh` session. This will add conda to your profile and it will be activated everytime you login.

```sh
source "[CONDA_PATH]/etc/profile.d/conda.sh"
conda init
```

For example,

```sh
source ~/miniconda/etc/profile.d/conda.sh
conda init
```

**`[CONDA_PATH]`** is the same as above.

`ssh` back into the remote computer

```sh
ssh -i "[PRIVATE_KEY_FILE_PATH]" [USERNAME]@[HOST_NAME]
```

Now you can verify your conda.

```sh
conda -V
```

You should get an output like `conda 4.8.3`.

## Create and activate a conda environment

`conda` activates its `base` python environment by default. If your run `python -V` in your remote machine, you will get the python version.

Now lets create a different conda environment, so that we can leave the base untouched.

```sh
conda create -y -n "[CONDA_ENV_NAME]" "python=[PYTHON_VERSION]"
```

**`[CONDA_ENV_NAME]`** is the name for your new conda environment and **`[PYTHON_VERSION]`** is the python version you need. You can go for something like `3.8`.

You can see a list of existing `conda` environments with following command:

```sh
conda info --envs
```

If you run this after creating the new conda environment you will see that and the `base` environment.

Run the following command to activate the conda environment, where **`[CONDA_ENV_NAME]`** is the name of the environment.

```
conda activate "[CONDA_ENV_NAME]"
    
```

You only need to create a `conda` environment once. There after you can activate it with the above command.

## Synchronise you project and data

We use[ **`rsync`**](https://en.wikipedia.org/wiki/Rsync) command to synchronize our project files. This only moves files that are new or have been changed. These steps should be run on your local machine.

Go to your project folder in the local machine.

```sh
cd PROJECT_FOLDER_PATH
```

Run `rsync`

```sh
rsync -zravuKLt --perms --executability -e "ssh -o  StrictHostKeyChecking=no -i [PRIVATE_KEY_FILE_PATH]" --exclude-from='[EXCLUDE_FILES_LIST_PATH]' ./ [USERNAME]@[HOST_NAME]:~/[FOLDER_NAME]/
    
```

Each character of  `-zravuKLt` is a flag. `z` is to compress files during the transfer, `r` is to find files recursively in the local path to transfer and `v` produces a verbose output. You can read more information about `rsync` and its options [here](https://linux.die.net/man/1/rsync).

`--perms` copies permission flags of the files to ther remote computer. Things like who can access the file.

`--executability` copies the executable-flag.

`-e "ssh -o StrictHostKeyChecking=no -i [PRIVATE_KEY_FILE_PATH]"` specified that it needs to connect to the server with a private key file. You only need this if you are specifying a private key file when you `ssh`. **`[PRIVATE_KEY_FILE_PATH]`** is same private key file that you used to `ssh` to your remote machine.

`--exclude-from='[EXCLUDE_FILES_LIST_PATH]'` specifies that come files should not be transferd. The list of file not to be transfered should be listed in file **`[EXCLUDE_FILES_LIST_PATH]`**.  Here's an example `exclude.txt` file:

```
 # exclude.txt

 .remote
 .git
 __pycache__
 .ipynb_checkpoints
 logs
 .DS_Store
 .*.swp
 *.egg-info/
 .idea
    
```

`./` is the source directory to be transferred and `[USERNAME]@[HOST_NAME]:~/[FOLDER_NAME]/`  is the destination.

## Install `pipenv` [optional]

If you want to use `pipenv` as your package manager, you can install it using `pip`. First `ssh` into your server and activate your conda environment. Then run the following command to install `pipenv`.

```sh
pip install pipenv
```

## Install `pip` packages from `requirements.txt` or `Pipfile`

First `ssh` into the remote machine and activate the conda environment.

Then, go to your project folder

```sh
cd FOLDER_NAME
```

Install packages from requirements.txt file.

```sh
pip install -r requirements.txt
```

If you use `pipenv`, you can just install from the `Pipfile`

```sh
pipenv install
```

## Run you Python script

`ssh` and activate the conda environment if you already haven't done so. Then go to you project folder.

If you installed packages with `pip` you can run your script directly.

```sh
python [PYTHON_SCRIPT_FILE]
```

And if you use `pipenv`,

```sh
pipenv run python [PYTHON_SCRIPT_FILE]
```

## Installing CUDA

You can install CUDA with conda but it sometimes doesn't work with deep speed.
In this case you need to install the CUDA version that matches the PyTorch or TensorFlow version.

You can see available versions with `apt-cache policy cuda`.
If it says `Unable to locate package cuda` you will have to download the
[packages from NVIDIA](https://developer.nvidia.com/cuda-toolkit-archive).
But before you install check the versions with `apt-cache policy cuda`.

Then install the respective versions; for example,

```bash
sudo apt-get install cuda=11.3.1-1
```

## [labml.ai remote](https://github.com/labmlai/labml/tree/master/remote)

This is still a tedious process with many commands. We have created an
[open-source library](https://github.com/labmlai/labml/tree/master/remote),
that does all of the above for you. You only need to provide a few configurations (project_name, hostname, private_key and username).

This is how it works,

```sh
pip install labml_remote

cd [PATH TO YOUR PROJECT FOLDER]

labml_remote --init # Give it project_name, SSH credentials etc

labml_remote python [PATH TO YOU PYTHON CODE] [ARGUMENTS TO YOUR  PYTHON CODE]
```
