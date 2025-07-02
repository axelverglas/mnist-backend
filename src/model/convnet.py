import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, input_size, n_kernels, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=n_kernels, kernel_size=5
            ),  # (28x28) → (24x24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (24x24) → (12x12)
            nn.Conv2d(
                in_channels=n_kernels, out_channels=n_kernels, kernel_size=5
            ),  # (12x12) → (8x8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (8x8) → (4x4)
            nn.Flatten(),  # (n_kernels x 4 x 4) = (n_kernels * 16)
            nn.Linear(n_kernels * 4 * 4, 50),
            nn.Linear(50, output_size),
        )

    def forward(self, x):
        return self.net(x)
