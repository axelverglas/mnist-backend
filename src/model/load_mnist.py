from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(
        datasets.MNIST("../data/raw", download=True, train=True, transform=tf),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST("../data/raw", download=True, train=False, transform=tf),
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, test_loader
