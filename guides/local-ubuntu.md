# Setting up Ubuntu with GPUs for machine learning

This is our setup for linux machines for machine learning.
Installing nvidia drivers became a lot easier with Ubuntu 20.04 and we use
 [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage
 [CUDA](https://developer.nvidia.com/cuda-zone) installations.
CUDA versions matter because PyTorch and Tensorflow support only older versions.

We use [terminator](https://terminator-gtk3.readthedocs.io/en/latest/) as the terminal,
and [oh-my-zsh](https://ohmyz.sh/) with [powerlevel10k theme](https://github.com/romkatv/powerlevel10k).
It gives indicators about the git branch, uncommitted changes, and conda environment on the shell.

![Z Shell](powerlevel10k.png)

Here's the setup procedure, up to installing PyTorch/Tensorflow.
This takes about an hour depending on your download speeds.

Upgrade packages.

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

I usually edit grub so that I can see the boot menu and troubleshoot (load ubuntu in recovery mode) if things go bad.
This was important before Ubuntu 20.04 when driver installations occasionally failed and I had to recover 
by going into recovery mode.
This step is not required.

Edit grub file

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

Install [terminator](https://terminator-gtk3.readthedocs.io/en/latest/).

```bash
sudo apt-get install terminator
```

Quit the terminal and start the terminator. You might want to add it to favorites for quick access.

Install Z Shell.

```bash
sudo apt-get install zsh
chsh -s $(which zsh)
```

Restart terminator, to see if it works.
In the zsh setup guide chose option (2) which will set defaults.

Install [oh-my-zsh](https://ohmyz.sh/).

```bash
sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
```

Restart terminator to see if it works.

Meslo fonts are needed by powerlevel10k theme. You need to download and install the fonts from here:
[https://github.com/romkatv/powerlevel10k#manual-font-installation]

Next change the font in terminator, in preferences.
1. Right-click on the terminator
2. Select Preferences 
3. Goto Profiles tab
4. Deselect *Use the system fixed-width font* 
5. Choose font MesloLGS Regular

Clone powerlevel10k into oh-my-zsh themes folder.

```bash
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
```

Edit `~/.zshrc` and set the theme to powerlevel10k.

```
ZSH_THEME="powerlevel10k/powerlevel10k"
```

Restart terminator for changes to take effect.

It will ask you to configure p10k. If this setup doesn't appear you can do it with `p10k configure`.

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

Next, we create a separate conda environment for our deep-learning work. You can have many environments.
See [miniconda guide](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
 to see details on how to manage conda environments.

Let's create an environment for our PyTorch work, and name it `myenv`.

```bash
conda create -n myenv "python=3.8"
```

If you add the option `-y`  it won't ask for confirmation.

Activate our new conda environment.

```bash
conda activate myenv
```

Let's install PyTorch 1.7 (latest version) with cuda.

```bash
conda install pytorch torchvision torchtext torchaudio cudatoolkit=11.0 -c pytorch
```

Install Jupyter notebooks and ipython.

```bash
conda install notebook -c conda-forge
```

You can check torch installation with `torch.cuda.device_count()`.
Run `ipython` and try the following,

```python
>>> import torch
>>> torch.cuda.is_available()
>>> torch.cuda.device_count()
```

We install *Tensorboard* if seperately if we don't install Tensorflow.

```bash
pip install tensorboard
```

You can install Tensorflow also through conda.

```bash
conda install tensorflow-gpu
```

I keep different environments for PyTorch and Tensorflow installations because they usually have different CUDA requirements.

You can check the tensorflow installation using `tf.config.list_physical_devices()`.

```python
>>> import tensorflow as tf
>>> tf.config.list_physical_devices()
```

This doesn't have anything to do with the computer setup, included it for anyone not familiar with creating RSA keys.

Create public/private keys with `ssh-keygen`:

```bash
ssh-keygen -t rsa -b 4096 -C "youremail@example.com"
```

Your public key will be located at `~/.ssh/id_rsa.pub` by default - `ssh-keygen` will ask you to confirm this path.
Copy the contents of the public key file to Github/Bitbucket SSH keys to access your repositories.
