"""
Лекция 5 — Пример 4: Классификация изображений с помощью предобученной модели ResNet18.
Демонстрирует Transfer Learning — использование модели, обученной на ImageNet (1000 классов).

Установка: pip install torch torchvision pillow
"""
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import os
import sys
from pathlib import Path

cwd = Path(__file__).parent.parent

# ============================================================
# 1. Загрузка предобученной модели ResNet18
# ============================================================
print("Загрузка модели ResNet18 (предобучена на ImageNet)...")
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Модель загружена! Параметров: {total_params:,}")

# ============================================================
# 2. Подготовка изображения
# ============================================================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],    # Стандартные значения ImageNet
        std=[0.229, 0.224, 0.225]
    ),
])

# Загружаем изображение
image_path = cwd / "images" / "dream_of.jpg"
if not os.path.exists(image_path):
    print(f"\nФайл '{image_path}' не найден.")
    print("Скачиваем тестовое изображение...")
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"
    urllib.request.urlretrieve(url, image_path)
    print(f"Сохранено: {image_path}")

img = Image.open(image_path).convert('RGB')
print(f"\nИзображение: {img.size[0]}×{img.size[1]} пикселей")

# Преобразуем для модели
input_tensor = preprocess(img).unsqueeze(0)  # Добавляем batch dimension
print(f"Тензор для модели: {input_tensor.shape}")  # [1, 3, 224, 224]

# ============================================================
# 3. Предсказание
# ============================================================
print("\nКлассификация...")
with torch.no_grad():
    output = model(input_tensor)

# Вероятности через Softmax
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# ============================================================
# 4. Загрузка названий классов ImageNet
# ============================================================
classes_file = "imagenet_classes.txt"
if not os.path.exists(classes_file):
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    urllib.request.urlretrieve(url, classes_file)

with open(classes_file) as f:
    categories = [line.strip() for line in f.readlines()]

print(f"Загружено {len(categories)} классов ImageNet")

# ============================================================
# 5. Результаты
# ============================================================
top5_prob, top5_idx = torch.topk(probabilities, 5)

print("\n" + "=" * 50)
print("Топ-5 предсказаний:")
print("=" * 50)
for i in range(5):
    category = categories[top5_idx[i]]
    probability = top5_prob[i].item() * 100
    bar = "█" * int(probability / 2)
    print(f"  {i+1}. {category:30s} {probability:6.2f}%  {bar}")

# ============================================================
# 6. Визуализация
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={'width_ratios': [1, 1.2]})

    # Изображение
    ax1.imshow(img)
    ax1.set_title("Входное изображение", fontsize=13)
    ax1.axis('off')

    # Гистограмма предсказаний
    names = [categories[idx] for idx in top5_idx]
    probs = [p.item() * 100 for p in top5_prob]
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(5)]

    bars = ax2.barh(range(4, -1, -1), probs, color=colors, edgecolor='white')
    ax2.set_yticks(range(4, -1, -1))
    ax2.set_yticklabels(names, fontsize=11)
    ax2.set_xlabel("Вероятность (%)", fontsize=12)
    ax2.set_title("Топ-5 предсказаний ResNet18", fontsize=13)

    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{prob:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig("classification_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nСохранено: classification_result.png")
except ImportError:
    print("\n(matplotlib не установлен — визуализация пропущена)")
