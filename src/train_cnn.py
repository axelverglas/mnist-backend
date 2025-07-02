import os
import torch
from model.convnet import ConvNet
from model.train import train, test
from model.load_mnist import get_dataloaders


def main():
    # Choix du device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Paramètres
    input_size = 28 * 28
    n_kernels = 6
    output_size = 10
    perm = torch.arange(0, 784).long()

    # Données
    train_loader, test_loader = get_dataloaders()

    # Modèle
    model = ConvNet(input_size, n_kernels, output_size).to(device)
    print(f"Parameters = {sum(p.numel() for p in model.parameters()) / 1e3:.3f}K")

    # Entraînement
    train(model, train_loader, device, perm=perm, n_epoch=1)

    # Évaluation
    test(model, test_loader, device, perm=perm)

    # Sauvegarde
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/mnist-0.0.1.pt")
    print("✅ Modèle CNN sauvegardé dans model/mnist-0.0.1.pt")


if __name__ == "__main__":
    main()
