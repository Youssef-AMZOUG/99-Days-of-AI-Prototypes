# Day 02 — Shortcut Learning Demonstration
# Spurious Correlation + Distribution Shift

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Custom Biased Dataset

class BiasedMNIST(datasets.MNIST):
    def __init__(self, root, train, transform=None, download=False, add_bias=True):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.add_bias = add_bias

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        if self.add_bias:
            image = self.add_spurious_patch(image, label)

        return image, label

    def add_spurious_patch(self, image, label):
        # image: tensor shape [1,28,28]
        image = image.clone()

        # deterministic color intensity based on label
        intensity = (label + 1) / 10.0

        # top-left 4x4 square
        image[:, 0:4, 0:4] = intensity

        return image


transform = transforms.ToTensor()

train_dataset = BiasedMNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True,
    add_bias=True
)

biased_test_dataset = BiasedMNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True,
    add_bias=True
)

clean_test_dataset = BiasedMNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True,
    add_bias=False
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
biased_test_loader = torch.utils.data.DataLoader(biased_test_dataset, batch_size=64, shuffle=False)
clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset, batch_size=64, shuffle=False)

# 2. CNN Model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training

for epoch in range(3):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 4. Evaluation Function

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


biased_acc = evaluate(biased_test_loader)
clean_acc = evaluate(clean_test_loader)

print(f"\nAccuracy on Biased Test Set: {biased_acc:.2f}%")
print(f"Accuracy on Clean Test Set:  {clean_acc:.2f}%")

# 5. Visualization Example

model.eval()
example_loader = torch.utils.data.DataLoader(clean_test_dataset, batch_size=1, shuffle=True)
image, label = next(iter(example_loader))
image = image.to(device)

with torch.no_grad():
    output = model(image)
    pred = output.argmax(dim=1)

plt.figure(figsize=(4,4))
plt.imshow(image.squeeze().cpu(), cmap="gray")
plt.title(f"Clean Image\nTrue: {label.item()} | Pred: {pred.item()}")
plt.axis("off")
plt.savefig("distribution_shift_example.png")
plt.show()
