"""Define models and generator functions which receives params as parameter, then add model to available models"""
import torch
import torch.nn as nn


class CustomBatchNorm(nn.Module):
    def __init__(self, channels, momentum=0.1):
        super(CustomBatchNorm, self).__init__()
        self.momentum = momentum
        self.first_pass = True
        self.register_buffer('running_mean', torch.zeros((channels)))
        self.register_buffer('running_var', torch.zeros((channels)))
        self.gamma = nn.Parameter(torch.nn.init.uniform_(torch.empty(1)))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.training:
            mean = x.transpose(0, 1).contiguous().view(x.transpose(0, 1).shape[0], -1).mean(1)
            var = x.transpose(0, 1).contiguous().view(x.transpose(0, 1).shape[0], -1).var(1)
            if self.first_pass:
                self.running_mean += mean.detach()
                self.running_var += var.detach()
                self.first_pass = False

            else:
                self.running_mean *= 1 - self.momentum
                self.running_mean += self.momentum * mean.detach()
                self.running_var *= 1 - self.momentum
                self.running_var += self.momentum * var.detach()

            mean = self.unsqueeze_m(mean, [0, 2, 3]).expand(x.shape[0], -1, x.shape[2], x.shape[3])
            var = self.unsqueeze_m(var, [0, 2, 3]).expand(x.shape[0], -1, x.shape[2], x.shape[3])

        else:
            mean = self.unsqueeze_m(self.running_mean, [0, 2, 3]).expand(x.shape[0], -1, x.shape[2], x.shape[3])
            var = self.unsqueeze_m(self.running_var, [0, 2, 3]).expand(x.shape[0], -1, x.shape[2], x.shape[3])

        return self.gamma * ((x - mean) / torch.clamp(var, min=1e-6)**0.5) + self.beta

    @staticmethod
    def unsqueeze_m(x, dims):
        for i in dims:
            x = torch.unsqueeze(x, i)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.path = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride),
            CustomBatchNorm(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            CustomBatchNorm(out_channels),
        )
        self.relu = nn.ReLU()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = None

    def forward(self, x):
        x_path = self.path(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return self.relu(x_path + x)


class CustomResNet(nn.Module):
    """CNN"""

    def __init__(self, params):
        super(CustomResNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding=(3, 3), stride=(2, 2)),
            nn.MaxPool2d((3, 3), stride=(2, 2))
        )
        self.block_64 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.block_128 = nn.Sequential(
            ResBlock(64, 128, 2),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
        )
        self.block_256 = nn.Sequential(
            ResBlock(128, 256, 2),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
        )
        self.block_512 = nn.Sequential(
            ResBlock(256, 512, 2),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )
        self.avg_pool = nn.AvgPool2d((7, 7))
        self.final = nn.Linear(512, params.num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.block_64(x)
        x = self.block_128(x)
        x = self.block_256(x)
        x = self.block_512(x)
        x = self.avg_pool(x)
        x = x.view(-1, 512)
        x = self.final(x)
        x = x.squeeze(1)
        return x


class VGG16(nn.Module):
    def __init__(self, params):
        super(VGG16, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            CustomBatchNorm(64),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            CustomBatchNorm(128),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            CustomBatchNorm(256),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            CustomBatchNorm(512),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            CustomBatchNorm(512),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc_1 = nn.Linear(7*7*512, 4096)
        self.fc_2 = nn.Linear(4096, 4096)
        self.fc_3 = nn.Linear(4096, params.num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = x.view(-1, 7*7*512)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = x.squeeze(1)
        return x


# Add all the models to the dictionary below. This is called in train using the desired model as input.
models = {
    '1': CustomResNet,
    '2': VGG16,
}
