"""
Лекция 5 — Пример 3: Классификация рукописных цифр (MNIST) с помощью CNN.
Использует PyTorch для обучения свёрточной нейронной сети.

Установка: pip install torch torchvision matplotlib
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ============================================================
# 1. Подготовка данных
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),                          # [0, 255] → [0.0, 1.0]
    transforms.Normalize((0.1307,), (0.3081,))     # Нормализация
])

# Загрузка MNIST (28×28, 10 цифр: 0–9)
print("Загрузка датасета MNIST...")
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

print(f"Обучающая выборка: {len(train_data)} изображений")
print(f"Тестовая выборка:  {len(test_data)} изображений")

# ============================================================
# 2. Визуализация данных
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_data[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"Цифра: {label}", fontsize=12)
    ax.axis('off')
plt.suptitle("Примеры из датасета MNIST", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("mnist_examples.png", dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 3. Определение модели CNN
# ============================================================
class DigitCNN(nn.Module):
    """
    Свёрточная нейронная сеть для распознавания цифр.

    Архитектура:
      Conv(1→32, 3×3) → ReLU → MaxPool(2×2) →
      Conv(32→64, 3×3) → ReLU → MaxPool(2×2) →
      Flatten → Linear(3136→128) → ReLU → Dropout → Linear(128→10)
    """
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28 → 28×28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → 14×14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 14×14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → 7×7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN().to(device)
print(f"\nМодель:\n{model}")
print(f"Устройство: {device}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Всего параметров: {total_params:,}")

# ============================================================
# 4. Обучение
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
train_losses = []
train_accuracies = []

print(f"\nОбучение ({epochs} эпох)...")
print("-" * 50)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f"  Эпоха {epoch+1}/{epochs}: loss = {avg_loss:.4f}, accuracy = {accuracy:.2f}%")

# ============================================================
# 5. Тестирование
# ============================================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"\n{'=' * 50}")
print(f"Точность на тестовых данных: {test_accuracy:.2f}%")
print(f"{'=' * 50}")

# ============================================================
# 6. Визуализация обучения
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, epochs + 1), train_losses, 'b-o', linewidth=2)
ax1.set_xlabel("Эпоха")
ax1.set_ylabel("Loss")
ax1.set_title("Функция потерь при обучении")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, epochs + 1), train_accuracies, 'g-o', linewidth=2)
ax2.set_xlabel("Эпоха")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Точность при обучении")
ax2.grid(True, alpha=0.3)

plt.suptitle("Процесс обучения CNN на MNIST", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("mnist_training.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 7. Предсказания на конкретных изображениях
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
model.eval()

for i, ax in enumerate(axes.flat):
    img, true_label = test_data[i]
    img_device = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_device)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    color = 'green' if predicted.item() == true_label else 'red'
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"Предсказание: {predicted.item()}\n"
                 f"({confidence.item()*100:.1f}%)",
                 fontsize=10, color=color)
    ax.axis('off')

plt.suptitle("Предсказания модели на тестовых данных", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("mnist_predictions.png", dpi=150, bbox_inches='tight')
plt.show()

# Сохранение модели
torch.save(model.state_dict(), "mnist_cnn.pth")
print("\nМодель сохранена: mnist_cnn.pth")
