import torch
from labml import monit
from labml.internal.analytics.models import ModelProbe
from labml.logger import inspect
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(1, 10)
        self.l2 = nn.Sequential(nn.Linear(10, 1), nn.Linear(1, 10))
        self.l3 = nn.Dropout(.1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


def test():
    with monit.section('Create model'):
        model = Model()
    with monit.section('Create probe'):
        probe = ModelProbe(model)

    with monit.section('Forward'):
        res = model(torch.tensor([0.1]))

    inspect(res)
    inspect(probe.forward_input)
    inspect(probe.forward_output.deep()['l*'].deep().get_dict())
    inspect(probe.parameters['*.bias'].get_dict())

    inspect(probe.backward_output)
    inspect(probe.backward_input)

    res.sum().backward()
    inspect(probe.backward_output.get_dict())
    inspect(probe.backward_input.get_dict())


if __name__ == '__main__':
    test()
