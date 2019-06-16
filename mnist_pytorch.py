"""
```trial
2019-06-16 13:15:04
Test
[[dirty]]: 📚 readme
start_step: 0
```
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from lab.experiment.pytorch import Experiment
from lab_globals import lab

# Declare the experiment
EXPERIMENT = Experiment(lab=lab,
                        name="mnist_pytorch",
                        python_file=__file__,
                        comment="Test",
                        check_repo_dirty=False
                        )

logger = EXPERIMENT.logger


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Add training loss to the logger.
        # The logger will queue the values and output the mean
        logger.store(train_loss=loss.item())

        # Print output to the console
        if batch_idx % args.log_interval == 0:
            # Reset the current line
            logger.clear_line(reset=True)
            # Print step
            logger.print_global_step(epoch * len(train_loader) + batch_idx)
            # Output the indicators without a newline
            # because we'll be replacing the same line
            logger.write(global_step=epoch * len(train_loader) + batch_idx, new_line=False)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Add test loss and accuracy to logger
    logger.store(test_loss=test_loss / len(test_loader.dataset))
    logger.store(accuracy=correct / len(test_loader.dataset))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    return parser.parse_args()


def main():
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Loading data
    with logger.monitor("Loading data"):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Model creation
    with logger.monitor("Create model"):
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Specify indicators
    logger.add_indicator("train_loss", queue_limit=10, is_print=True)
    logger.add_indicator("test_loss", is_histogram=False, is_print=True)
    logger.add_indicator("accuracy", is_histogram=False, is_print=True)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.add_indicator(name, is_histogram=True, is_print=False)
            logger.add_indicator(f"{name}_grad", is_histogram=True, is_print=False)

    # Start the experiment
    EXPERIMENT.start_train(0)

    # Create a monitored iterator
    monitor = logger.iterator(range(0, args.epochs))

    # Loop through the monitored iterator
    for epoch in monitor:
        global_step = (epoch + 1) * len(train_loader)

        # Delayed keyboard interrupt handling to use
        # keyboard interrupts to end the loop.
        # This will capture interrupts and finish
        # the loop at the end of processing the iteration;
        # i.e. the loop won't stop in the middle of an epoch.
        try:
            with logger.delayed_keyboard_interrupt():

                # Training and testing
                train(args, model, device, train_loader, optimizer, epoch)
                test(model, device, test_loader)

                # Add histograms with model parameter values and gradients
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.store(name, param.data.cpu().numpy())
                        logger.store(f"{name}_grad", param.grad.cpu().numpy())

                # Output the progress summaries to `trial.yaml` and
                # to the python file header
                progress_dict = logger.get_progress_dict(
                    global_step=global_step)
                EXPERIMENT.save_progress(progress_dict)

                # Clear line and output to console
                logger.clear_line(reset=True)
                logger.print_global_step(global_step)
                logger.write(global_step=global_step, new_line=False)

                # Output the progress of the loop,
                # time taken and time remaining
                monitor.progress()

                # Clear line and go to the next line;
                # that is, we add a new line to the output
                # at the end of each epoch
                logger.clear_line(reset=False)

        # Handled delayed interrupt
        except KeyboardInterrupt:
            logger.clear_line(reset=False)
            logger.log("\nKilling loop...")
            break

    logger.clear_line(reset=False)


if __name__ == '__main__':
    main()
