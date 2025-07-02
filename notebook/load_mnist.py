import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Détection du device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
print(f"Using device: {device}")

# Définition des transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Chargement du dataset
train_dataset = datasets.MNIST("../data/raw", download=True, train=True, transform=transform)

# Trouver une image pour chaque chiffre de 0 à 9
images = []
labels = []
found_digits = set()

for img, label in train_dataset:
    if label not in found_digits:
        images.append(img)
        labels.append(label)
        found_digits.add(label)
    if len(found_digits) == 10:
        break

# Affichage des 10 chiffres dans l'ordre
fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for i in range(10):
    axes[i].imshow(images[i].squeeze(), cmap='gray')
    axes[i].set_title(f'Label: {labels[i]}')
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png')
plt.close()
print("Images sauvegardées dans mnist_samples.png")