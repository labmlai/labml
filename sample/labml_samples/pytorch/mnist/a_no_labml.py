import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms

from labml import lab


class Net(nn.Module):
    def __init__(self):
        super().__init__()
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
        return self.fc2(x)


def train(epoch, model, optimizer, train_loader, device, train_log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % train_log_interval == 0:
            print(f'train epoch: {epoch}'
                  f' [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]'
                  f'\tLoss: {loss.item():.6f}')


def validate(epoch, model, valid_loader, device):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            valid_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= len(valid_loader.dataset)
    valid_accuracy = 100. * correct / len(valid_loader.dataset)

    print(f'\nTest set: Average loss: {valid_loss:.4f},'
          f' Accuracy: {correct}/{len(valid_loader.dataset)}'
          f' ({valid_accuracy:.0f}%)\n')


def main():
    epochs = 10

    is_save_models = True
    train_batch_size = 64
    valid_batch_size = 1000

    use_cuda = True
    seed = 5
    train_log_interval = 10

    learning_rate = 0.01

    # get device
    is_cuda = use_cuda and torch.cuda.is_available()
    if not is_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:0")

    # data transform
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # train loader
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(str(lab.get_data_path()),
                       train=True,
                       download=True,
                       transform=data_transform),
        batch_size=train_batch_size, shuffle=True)

    # valid loader
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(str(lab.get_data_path()),
                       train=False,
                       download=True,
                       transform=data_transform),
        batch_size=valid_batch_size, shuffle=False)

    # model
    model = Net().to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # set seeds
    torch.manual_seed(seed)

    # training loop
    for epoch in range(1, epochs + 1):
        train(epoch, model, optimizer, train_loader, device, train_log_interval)
        validate(epoch, model, valid_loader, device)

    if is_save_models:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
