"""
Modified from https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/01-mnist-hello-world.ipynb
Added labml logger
"""
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from labml import lab, experiment
from labml.utils.lightning import LabMLLightningLogger


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def main():
    # Init our model
    mnist_model = MNISTModel()

    # Init DataLoader from MNIST Dataset
    train_ds = MNIST(str(lab.get_data_path()), train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=32)

    # Initialize a trainer
    trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20, logger=LabMLLightningLogger())

    # Train the model ⚡
    with experiment.record(name='mnist_lightening', disable_screen=True):
        trainer.fit(mnist_model, train_loader)


if __name__ == '__main__':
    main()
