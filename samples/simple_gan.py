import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np

from lab import logger, configs
from lab import training_loop
from lab.experiment.pytorch import Experiment

import matplotlib.pyplot as plt

from lab.logger.artifacts import Image, IndexedText

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def noise(device, bs, dim):
    """Generate random Gaussian noise.

    Inputs:
    - bs: integer giving the batch size of noise to generate.
    - dim: integer giving the dimension of the the noise to generate.

    Returns:
    A PyTorch Tensor containing Gaussian noise with shape [bs, dim]
    """

    return (torch.randn((bs, dim))).to(device)


bce_loss = nn.BCEWithLogitsLoss()


def DLoss(logits_real, logits_fake, targets_real, targets_fake):
    """
    d1 - binary cross entropy loss between outputs of the Discriminator with real images
         (logits_real) and targets_real.

    d2 - binary cross entropy loss between outputs of the Discriminator with the generated fake images
         (logits_fake) and targets_fake.
    """
    d1 = bce_loss(logits_real, targets_real)
    d2 = bce_loss(logits_fake, targets_fake)

    return d1 + d2


def GLoss(logits_fake, targets_real):
    """
    The aim of the Generator is to fool the Discriminator into "thinking" the generated images are real.

    g_loss - binary cross entropy loss between the outputs of the Discriminator with the generated fake images
         (logits_fake) and targets_real.

    Thus, the gradients estimated with the above loss corresponds to generator producing fake images that
    fool the discriminator.
    """
    return bce_loss(logits_fake, targets_real)


class Generator(nn.Module):
    def __init__(self, noise_dim=100, out_size=784):
        super(Generator, self).__init__()

        self.layer1 = nn.Linear(noise_dim, 256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1024)
        self.layer4 = nn.Linear(1024, out_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        layers = nn.Sequential(
            self.layer1,
            self.leaky_relu,
            self.layer2,
            self.leaky_relu,
            self.layer3,
            self.leaky_relu,
            self.layer4,
            self.tanh
        )

        x = layers(x)
        n, i = x.size()
        x = x.view(n, 1, 28, 28)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_size=784):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Linear(input_size, 512)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(x.size(0), -1)

        layers = nn.Sequential(
            self.layer1,
            self.leaky_relu,
            self.layer2,
            self.leaky_relu,
            self.layer3,
        )
        return layers(x)


class GAN:
    def __init__(self, c: 'Configs'):
        self.batch_size = c.batch_size
        self.epochs = c.epochs
        self.noise_dim = c.noise_dim

        self.device = c.device

        self.generator = c.generator
        self.discriminator = c.discriminator
        self.optimizer_G = c.optimizer_G
        self.optimizer_D = c.optimizer_D

        self.train_loader = c.train_loader
        self.test_loader = c.test_loader

        self.train_log_interval = c.train_log_interval
        self.loop = c.training_loop
        self.__is_log_parameters = c.is_log_parameters

        self.image_directory = c.image_directory

    def _train(self, epoch):
        for i, (images, _) in logger.enum("Train", self.train_loader):
            targets_real = torch.empty(images.size(0), 1, device=self.device).uniform_(0.8, 1.0)
            targets_fake = torch.empty(images.size(0), 1, device=self.device).uniform_(0.0, 0.2)

            images = images.to(self.device)

            self.optimizer_D.zero_grad()
            logits_real = self.discriminator(images)
            fake_images = self.generator(
                noise(self.device, self.batch_size, self.noise_dim)).detach()
            logits_fake = self.discriminator(fake_images)
            discriminator_loss = DLoss(logits_real, logits_fake, targets_real, targets_fake)
            discriminator_loss.backward()
            self.optimizer_D.step()

            self.optimizer_G.zero_grad()
            fake_images = self.generator(noise(self.device, self.batch_size, self.noise_dim))
            logits_fake = self.discriminator(fake_images)
            generator_loss = GLoss(logits_fake, targets_real)
            generator_loss.backward()
            self.optimizer_G.step()

            logger.store(G_Loss=generator_loss.item())
            logger.store(D_Loss=discriminator_loss.item())
            logger.add_global_step()

        for j in range(1, 10):
            img = fake_images[j].squeeze()
            logger.store('generated', img)

    def __call__(self):
        logger.add_artifact(Image('generated'))
        for i in self.loop:
            self._train(i)


class LoaderConfigs(configs.Configs):
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader


class Configs(training_loop.TrainingLoopConfigs, LoaderConfigs):
    epochs: int = 50

    loop_step = 'loop_step'
    loop_count = 'loop_count'

    is_save_models = True
    batch_size: int = 128
    test_batch_size: int = 128

    use_cuda: float = True
    cuda_device: int = 0
    seed: int = 5
    train_log_interval: int = 10

    is_log_parameters: bool = True

    device: any

    generator: nn.Module
    discriminator: nn.Module

    optimizer_G: optim
    optimizer_D: optim
    lr_G: float = 0.0002
    beta_G: tuple = (0.5, 0.999)
    lr_D: float = 0.0002
    beta_D: tuple = (0.5, 0.999)

    noise_dim: int = 100

    set_seed = 'set_seed'

    main: GAN

    image_directory: str = "images/{}.png"


@Configs.calc(Configs.generator)
def set_generator(c: Configs):
    return Generator().to(c.device)


@Configs.calc(Configs.discriminator)
def set_discriminator(c: Configs):
    return Discriminator().to(c.device)


@Configs.calc(Configs.optimizer_G)
def generator_optimizer(c: Configs):
    return optim.Adam(c.generator.parameters(), lr=c.lr_G, betas=c.beta_G)


@Configs.calc(Configs.optimizer_D)
def discriminator_optimizer(c: Configs):
    return optim.Adam(c.discriminator.parameters(), lr=c.lr_D, betas=c.beta_D)


@Configs.calc(Configs.set_seed)
def set_seed(c: Configs):
    torch.manual_seed(c.seed)


@Configs.calc(Configs.loop_count)
def loop_count(c: Configs):
    return c.epochs * len(c.train_loader)


@Configs.calc(Configs.loop_step)
def loop_step(c: Configs):
    return len(c.train_loader)


@Configs.calc(Configs.device)
def device(*, use_cuda, cuda_device):
    from lab.util.pytorch import get_device

    return get_device(use_cuda, cuda_device)


def _data_loader(is_train, batch_size):
    return torch.utils.data.DataLoader(
        datasets.MNIST(str(logger.get_data_path()),
                       train=is_train,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.5], std=[0.5])
                       ])),
        batch_size=batch_size, shuffle=True, drop_last=True)


@Configs.calc([Configs.train_loader, Configs.test_loader])
def data_loaders(c: Configs):
    train = _data_loader(True, c.batch_size)
    test = _data_loader(False, c.test_batch_size)

    return train, test


def main():
    conf = Configs()
    experiment = Experiment(writers={'sqlite'})
    experiment.calc_configs(conf,
                            {},
                            run_order=['set_seed', 'main'])
    experiment.start()
    conf.main()


if __name__ == '__main__':
    main()
