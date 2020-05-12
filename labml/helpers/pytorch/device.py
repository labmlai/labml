import torch

from labml.configs import BaseConfigs


class DeviceInfo:
    def __init__(self, *,
                 use_cuda: bool,
                 cuda_device: int):
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device
        self.cuda_count = torch.cuda.device_count()

        self.is_cuda = self.use_cuda and torch.cuda.is_available()
        if not self.is_cuda:
            self.device = torch.device('cpu')
        else:
            if self.cuda_device < self.cuda_count:
                self.device = torch.device('cuda', self.cuda_device)
            else:
                self.device = torch.device('cuda', self.cuda_count - 1)

    def __str__(self):
        if not self.is_cuda:
            return "CPU"

        if self.cuda_device < self.cuda_count:
            return f"GPU:{self.cuda_device} - {torch.cuda.get_device_name(self.cuda_device)}"
        else:
            return (f"GPU:{self.cuda_count - 1}({self.cuda_device}) "
                    f"- {torch.cuda.get_device_name(self.cuda_count - 1)}")


class DeviceConfigs(BaseConfigs):
    cuda_device: int = 0
    use_cuda: bool = True

    device_info: DeviceInfo

    device: torch.device


@DeviceConfigs.calc(DeviceConfigs.device)
def _device(c: DeviceConfigs):
    return c.device_info.device


DeviceConfigs.set_hyperparams(DeviceConfigs.cuda_device, DeviceConfigs.use_cuda,
                              is_hyperparam=False)


@DeviceConfigs.calc(DeviceConfigs.device_info)
def _device_info(c: DeviceConfigs):
    return DeviceInfo(use_cuda=c.use_cuda,
                      cuda_device=c.cuda_device)
