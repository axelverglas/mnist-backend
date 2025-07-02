import torch
import torch.nn.functional as F


def train(model, train_loader, device, perm=None, n_epoch=1):
    if perm is None:
        perm = torch.arange(0, 784).long()  # Permutation par défaut (identité)

    model.train()  # Mode entraînement
    optimizer = torch.optim.AdamW(model.parameters())  # Optimiseur

    for epoch in range(n_epoch):
        for step, (data, target) in enumerate(train_loader):
            # Envoi vers GPU / MPS / CPU
            data, target = data.to(device), target.to(device)

            # Application de la permutation Toeplitz (désordre spatial)
            data = data.view(-1, 28 * 28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)

            # Étapes d'entraînement : Forward, Loss, Backward, Optimizer step
            optimizer.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            # Log toutes les 100 itérations
            if step % 100 == 0:
                print(f"epoch={epoch}, step={step}: train loss={loss.item():.4f}")


def test(model, test_loader, device, perm=None):
    if perm is None:
        perm = torch.arange(0, 784).long()

    model.eval()  # Mode évaluation

    test_loss = 0
    correct = 0

    with torch.no_grad():  # Pas de calcul de gradient pendant le test
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Appliquer la permutation Toeplitz
            data = data.view(-1, 28 * 28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)

            # Prédiction
            logits = model(data)

            # Calcul du loss cumulé
            test_loss += F.cross_entropy(logits, target, reduction="sum").item()

            # Prédictions finales
            pred = torch.argmax(logits, dim=1)

            # Comptage des bonnes prédictions
            correct += (pred == target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print(f"test loss={test_loss:.4f}, accuracy={accuracy:.4f}")
