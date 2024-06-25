# Setting up Ubuntu with GPUs for machine learning

This is our setup for linux machines for machine learning.
Installing nvidia drivers became a lot easier with Ubuntu 20.04 onwards and we use
 [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage
 [CUDA](https://developer.nvidia.com/cuda-zone) installations.
CUDA versions matter because PyTorch and Tensorflow support only older versions.

We use [terminator](https://terminator-gtk3.readthedocs.io/en/latest/) as the terminal,
and [oh-my-zsh](https://ohmyz.sh/).
It gives indicators about the git branch, uncommitted changes, and conda environment on the shell.

![Z Shell](powerlevel10k.png)

Here's the setup procedure, up to installing PyTorch.
This takes about an hour depending on your download speeds.

```bash
sudo apt-get update
sudo apt-get upgrade
```

If you ran this on a fresh Ubuntu 20.04 installation it would upgrade the kernel and you will be required to restart the computer.

Install **vim** and **git**.

```bash
sudo apt-get install vim-gtk3
sudo apt-get install git gitk
```

### Edit grub file \[Optional\]

I usually edit grub so that I can see the boot menu and troubleshoot (load ubuntu in recovery mode) if things go bad.
This was important before Ubuntu 20.04 when driver installations occasionally failed and I had to recover 
by going into recovery mode.
This step is not required.

```bash
sudo vim /etc/default/grub
```

Update the following settings. This gives a 5-second boot menu, and shows all the ubuntu startup logs instead of the nice splash screen.

```cfg
GRUB_TIMEOUT=5 # Wait 5 seconds
GRUB_TIMEOUT_STYLE=menu # Display the menu 
GRUB_CMDLINE_LINUX_DEFAULT = '' # remove quiet and splash so that you can see details at bootup
```

Update grub for changes to take effect.

```bash
sudo update-grub
```

You can restart the computer to see if this worked.

### Install Drivers

Next, we want to install the nvidia drivers. You can check if you already have them installed by running `nvidia-smi`.

```bash
sudo apt-get install --no-install-recommends nvidia-driver-450 
```

`--no-install-recommends` doesn't install packages recommended as dependencies.
Only required dependencies will be installed.
This does a lean driver installation.
You can install a version higher than `450` if available.
You should restart the computer for the driver installation to take effect.

Run the following after the restart to make sure drivers are installed.
It should show the driver version and a list of GPUs on the computer.

```bash
nvidia-smi
```

### Install [terminator](https://terminator-gtk3.readthedocs.io/en/latest/).

```bash
sudo apt-get install terminator
```

Quit the terminal and start the terminator. You might want to add it to favorites for quick access.

### Install Z Shell and [oh-my-zsh](https://ohmyz.sh/)

```bash
sudo apt-get install zsh
sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
```

Restart terminator to see if it works.

### Install miniconda

Now that shell is setup we install miniconda.
Miniconda lets us manage different python environments - even with different versions of python.
It also takes care of cuda versions.

Download miniconda.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
```

Install miniconda to `~/miniconda`.

```bash
bash miniconda.sh -b -p ~/miniconda
```

Next, we need to set conda paths on oh-my-zsh. We do this by adding an `oh-my-zsh` plugin.
Create a file `~/.oh-my-zsh/conda.zsh` and include this in it.

```bash
source ~/miniconda/etc/profile.d/conda.sh
```

Restart terminator again for it to take effect. You can check conda installation by running `conda -V`.

### Create a conda environment

Next, we create a separate conda environment for our deep-learning work. You can have many environments.
See [miniconda guide](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
 for details on how to manage conda environments.

Let's create an environment for our PyTorch work, and name it `myenv`.

```bash
conda create -n myenv "python=3.8"
```

If you add the option `-y`  it won't ask for confirmation.

Activate our new conda environment.

```bash
conda activate myenv
```

### Install PyTorch and other packages

Let's install PyTorch latest version with cuda.

```bash
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install other packages like Jupyter Notebooks.

```bash
pip install setuptools numpy opt_einsum sentencepiece jupyterlab ninja einsum ipywidgets matplotlib seaborn
```

You can check torch installation with `torch.cuda.device_count()`.
Run `ipython` and try the following,

```python
>>> import torch
>>> torch.cuda.is_available()
>>> torch.cuda.device_count()
```

### Create SSH Keys

Create public/private keys with `ssh-keygen`:

```bash
ssh-keygen -t rsa -b 4096 -C "youremail@example.com"
```

Your public key will be located at `~/.ssh/id_rsa.pub` by default - `ssh-keygen` will ask you to confirm this path.
Copy the contents of the public key file to Github/Bitbucket SSH keys to access your repositories.
