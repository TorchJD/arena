from torch import Tensor, nn


class Cifar10Model(nn.Module):
    N_CLASSES = 10

    def __init__(self, activation_function: type[nn.Module] = nn.ELU):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            activation_function(),
            nn.Conv2d(32, 64, 3, groups=32),
            nn.MaxPool2d(2),
            activation_function(),
            nn.Conv2d(64, 64, 3, groups=64),
            nn.MaxPool2d(3),
            activation_function(),
            nn.Flatten(),
            nn.Linear(1024, 128),
            activation_function(),
            nn.Linear(128, self.N_CLASSES),
            nn.Flatten(start_dim=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)
