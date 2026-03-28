# Обработка изображений с помощью нейронных сетей

---

## Содержание

1. [Введение: почему нейронные сети?](#1-введение-почему-нейронные-сети)
2. [Основы нейронных сетей](#2-основы-нейронных-сетей)
3. [Свёрточные нейронные сети (CNN)](#3-свёрточные-нейронные-сети-cnn)
4. [Практика: классификация изображений](#4-практика-классификация-изображений)
5. [Детекция объектов (Object Detection)](#5-детекция-объектов-object-detection)
6. [Сегментация изображений](#6-сегментация-изображений)
7. [Генерация и улучшение изображений](#7-генерация-и-улучшение-изображений)
8. [Перенос стиля (Style Transfer)](#8-перенос-стиля-style-transfer)
9. [Контрольные вопросы](#9-контрольные-вопросы)
10. [Список литературы и ресурсов](#10-список-литературы-и-ресурсов)

---

## 1. Введение: почему нейронные сети?

### 1.1. Ограничения классических методов

В предыдущих лекциях мы изучили **классические** методы обработки изображений: размытие, резкость, пороговая бинаризация, детекция границ. Эти методы хорошо работают для **простых** задач, но имеют ограничения:

| Задача | Классический подход | Проблема |
|---|---|---|
| Распознавание лиц | Каскады Хаара | Чувствительны к повороту, освещению |
| Классификация объектов | Ручные признаки (HOG, SIFT) | Требует ручного выбора признаков |
| Сегментация сцены | Пороговые методы | Не понимают контекст изображения |
| Генерация изображений | — | Невозможно классическими методами |

### 1.2. Нейронные сети — универсальный инструмент

**Нейронные сети** автоматически извлекают **признаки** из данных, без необходимости вручную их программировать:

```
Классический подход:
  Изображение → [Ручные признаки] → [Классификатор] → Результат
                 (HOG, SIFT, LBP)    (SVM, kNN)

Нейросетевой подход:
  Изображение → [Нейронная сеть] → Результат
                 (сама извлекает      (класс, маска,
                  признаки)            координаты)
```

### 1.3. Краткая история

| Год | Событие | Значимость |
|---|---|---|
| 1943 | Модель нейрона Маккаллока–Питтса | Первая математическая модель нейрона |
| 1986 | Алгоритм обратного распространения ошибки | Эффективное обучение многослойных сетей |
| 1998 | LeNet-5 (Ян Лекун) | Первая CNN для распознавания цифр |
| 2012 | AlexNet (ImageNet) | Прорыв: CNN обогнали классические методы |
| 2014 | GAN (Гудфеллоу) | Генерация реалистичных изображений |
| 2015 | ResNet (152 слоя) | Глубокие сети без затухания градиента |
| 2020+ | Vision Transformer (ViT) | Трансформеры для изображений |

---

## 2. Основы нейронных сетей

### 2.1. Искусственный нейрон

Искусственный нейрон — это математическая функция, которая принимает несколько входов, умножает каждый на **вес**, суммирует и применяет **функцию активации**:

```
         Входы      Веса
         x₁ ──── w₁ ───┐
                        │
         x₂ ──── w₂ ───┼──→ Σ(xᵢ × wᵢ) + b ──→ f(z) ──→ выход (y)
                        │
         x₃ ──── w₃ ───┘
                              b — смещение (bias)
                              f — функция активации
```

**Формула:**

```
z = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b
y = f(z)
```

### 2.2. Функции активации

Функция активации вносит **нелинейность**, позволяя сети обучаться сложным зависимостям:

| Функция | Формула | Диапазон | Применение |
|---|---|---|---|
| **Sigmoid** | σ(z) = 1 / (1 + e⁻ᶻ) | (0, 1) | Бинарная классификация |
| **ReLU** | f(z) = max(0, z) | [0, +∞) | Скрытые слои (самая популярная) |
| **Tanh** | tanh(z) | (-1, 1) | Нормализованный вывод |
| **Softmax** | eᶻⁱ / Σeᶻʲ | (0, 1), сумма = 1 | Многоклассовая классификация |

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# Функции активации
sigmoid = 1 / (1 + np.exp(-x))
relu = np.maximum(0, x)
tanh = np.tanh(x)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, y, name in zip(axes, [sigmoid, relu, tanh], ['Sigmoid', 'ReLU', 'Tanh']):
    ax.plot(x, y, linewidth=2)
    ax.set_title(name, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()
```

### 2.3. Архитектура многослойной сети

```
Входной слой      Скрытые слои       Выходной слой
(Input Layer)     (Hidden Layers)    (Output Layer)

  [ x₁ ]           [ h₁ ]  [ h₁ ]      [ y₁ ]
  [ x₂ ]  ──────→  [ h₂ ]  [ h₂ ]  ──→ [ y₂ ]
  [ x₃ ]           [ h₃ ]  [ h₃ ]      [ y₃ ]
  [ x₄ ]           [ h₄ ]

  Пиксели         Абстрактные        Классы
  изображения     признаки           (кот, собака...)
```

### 2.4. Обучение нейронной сети

Обучение — это процесс подбора весов для минимизации **функции потерь** (loss function):

```
1. Прямой проход (Forward Pass):
   Входные данные → через все слои → предсказание

2. Вычисление ошибки (Loss):
   L = f(предсказание, правильный_ответ)

3. Обратное распространение (Backpropagation):
   Вычисление градиентов ∂L/∂w для каждого веса

4. Обновление весов (Gradient Descent):
   w_new = w_old - learning_rate × ∂L/∂w

5. Повторяем шаги 1–4 много раз (эпох)
```

### 2.5. Простейшая нейронная сеть на Python (без библиотек)

```python
import numpy as np

# Простая нейронная сеть с 1 скрытым слоем для XOR
# Входы и правильные ответы
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Инициализация весов случайными значениями
np.random.seed(42)
weights_hidden = np.random.randn(2, 4) * 0.5   # 2 входа → 4 нейрона
bias_hidden = np.zeros((1, 4))
weights_output = np.random.randn(4, 1) * 0.5   # 4 нейрона → 1 выход
bias_output = np.zeros((1, 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

learning_rate = 1.0

# Обучение: 10 000 итераций
for epoch in range(10000):
    # --- Прямой проход ---
    hidden = sigmoid(X @ weights_hidden + bias_hidden)
    output = sigmoid(hidden @ weights_output + bias_output)

    # --- Ошибка ---
    error = y - output

    # --- Обратное распространение ---
    d_output = error * sigmoid_derivative(output)
    d_hidden = (d_output @ weights_output.T) * sigmoid_derivative(hidden)

    # --- Обновление весов ---
    weights_output += hidden.T @ d_output * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_hidden += X.T @ d_hidden * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if epoch % 2000 == 0:
        loss = np.mean(error ** 2)
        print(f"Эпоха {epoch}, loss = {loss:.6f}")

# Результат
print("\nПредсказания после обучения:")
for i in range(4):
    print(f"  {X[i]} → {output[i][0]:.4f}  (ожидали {y[i][0]})")
```

---

## 3. Свёрточные нейронные сети (CNN)

### 3.1. Почему обычные сети плохо работают с изображениями?

Полносвязная сеть для изображения 224×224×3 имеет **150 528 входов**. Если первый скрытый слой содержит 1000 нейронов — это **150 миллионов** весов только в первом слое! Это приводит к:

- Огромному количеству параметров
- Переобучению
- Потере пространственной информации

### 3.2. Идея свёрточных сетей

CNN используют **свёрточные слои** вместо полносвязных. Вспомним фильтры из лекции 4 — свёрточная сеть **обучает ядра (фильтры) автоматически**:

```
Классические фильтры:         CNN:
Ядро задаётся вручную          Ядро обучается из данных

┌───┬───┬───┐                  ┌───┬───┬───┐
│ 1 │ 1 │ 1 │  ← Box Blur     │ ? │ ? │ ? │  ← Сеть сама
├───┼───┼───┤  (мы задали)     ├───┼───┼───┤    подберёт
│ 1 │ 1 │ 1 │                  │ ? │ ? │ ? │    значения
├───┼───┼───┤                  ├───┼───┼───┤
│ 1 │ 1 │ 1 │                  │ ? │ ? │ ? │
└───┴───┴───┘                  └───┴───┴───┘
```

### 3.3. Архитектура CNN

Типичная CNN состоит из нескольких типов слоёв:

```
Вход         Conv+ReLU      Pool      Conv+ReLU      Pool      FC       Выход
(224×224×3)  (222×222×32)  (111×111)  (109×109×64)  (54×54)  (1000)   (10 классов)

┌────────┐   ┌────────┐   ┌──────┐   ┌────────┐   ┌──────┐  ┌─────┐  ┌─────┐
│        │   │Выделение│   │Умень-│   │Сложные│    │Умень-│  │Клас-│  │Кот  │
│ Изобр. │──→│ линий,  │──→│шение │──→│формы, │──→│шение │──→│сифи-│──→│Соба-│
│        │   │ краёв   │   │размер│   │текстур│   │размер│  │кация│  │ка   │
└────────┘   └────────┘   └──────┘   └────────┘   └──────┘  └─────┘  └─────┘
```

### 3.4. Слои CNN

#### Свёрточный слой (Convolution Layer)

Применяет набор обучаемых фильтров к входному изображению:

```
Вход (5×5×1)     Фильтр (3×3)     Карта признаков (3×3)

┌─┬─┬─┬─┬─┐     ┌──┬──┬──┐      ┌──┬──┬──┐
│1│0│1│0│1│     │ 1│ 0│-1│      │ 1│-1│ 0│
├─┼─┼─┼─┼─┤     ├──┼──┼──┤      ├──┼──┼──┤
│0│1│0│1│0│  *  │ 0│ 1│ 0│  =   │ 2│ 0│-2│
├─┼─┼─┼─┼─┤     ├──┼──┼──┤      ├──┼──┼──┤
│1│0│1│0│1│     │-1│ 0│ 1│      │-1│ 1│ 0│
├─┼─┼─┼─┼─┤     └──┴──┴──┘      └──┴──┴──┘
│0│1│0│1│0│
├─┼─┼─┼─┼─┤     Stride = 1
│1│0│1│0│1│     Padding = 0
└─┴─┴─┴─┴─┘
```

**Ключевые параметры:**

| Параметр | Описание | Типичное значение |
|---|---|---|
| **Kernel size** | Размер фильтра | 3×3, 5×5 |
| **Stride** | Шаг перемещения фильтра | 1, 2 |
| **Padding** | Добавление нулей по краям | 0, 1 |
| **Filters** | Количество фильтров | 32, 64, 128 |

#### Слой подвыборки (Pooling Layer)

Уменьшает размер карты признаков, сохраняя важные характеристики:

```
Max Pooling 2×2:

┌─┬─┬─┬─┐        ┌─┬─┐
│1│3│2│1│        │5│3│
├─┼─┼─┼─┤   →   ├─┼─┤
│5│2│0│3│        │8│4│
├─┼─┼─┼─┤        └─┴─┘
│4│1│8│2│
├─┼─┼─┼─┤   max(1,3,5,2)=5  max(2,1,0,3)=3
│2│0│3│4│   max(4,1,2,0)=4  max(8,2,3,4)=8
└─┴─┴─┴─┘
```

### 3.5. Что «видит» каждый слой CNN?

По мере углубления сеть распознаёт всё более сложные признаки:

```
Слой 1: Линии, грани, простые текстуры
         ───  │  ╱  ╲  ═══

Слой 2: Углы, кривые, комбинации линий
         ┌─  ─┐  ◠  ◡  ∠

Слой 3: Части объектов (глаза, колёса, лапы)
         👁  ◎  🐾

Слой 4+: Целые объекты и сцены
         🐱  🚗  🏠
```

---

## 4. Практика: классификация изображений

### 4.1. Установка необходимых библиотек

```bash
pip install torch torchvision matplotlib pillow
```

### 4.2. Классификация рукописных цифр (MNIST) — с нуля

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- 1. Подготовка данных ---
# Преобразования: в тензор + нормализация
transform = transforms.Compose([
    transforms.ToTensor(),                          # [0, 255] → [0.0, 1.0]
    transforms.Normalize((0.1307,), (0.3081,))     # Нормализация (среднее, std)
])

# Загрузка датасета MNIST (рукописные цифры 28×28)
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# --- 2. Визуализация данных ---
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_data[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"Цифра: {label}", fontsize=12)
    ax.axis('off')
plt.suptitle("Примеры из MNIST", fontsize=14)
plt.tight_layout()
plt.show()

# --- 3. Определение модели CNN ---
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Свёрточные слои
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28×1 → 28×28×32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → 14×14×32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 14×14×64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → 7×7×64
        )
        # Полносвязные слои
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                                  # → 7×7×64 = 3136
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),                              # Регуляризация
            nn.Linear(128, 10)                             # 10 классов (цифры 0–9)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = DigitCNN()
print(model)

# --- 4. Обучение ---
criterion = nn.CrossEntropyLoss()     # Функция потерь
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()              # Обнуляем градиенты
        outputs = model(images)            # Прямой проход
        loss = criterion(outputs, labels)  # Вычисляем потери
        loss.backward()                    # Обратное распространение
        optimizer.step()                   # Обновляем веса

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    train_losses.append(avg_loss)
    print(f"Эпоха {epoch+1}/{epochs}: loss = {avg_loss:.4f}, accuracy = {accuracy:.2f}%")

# --- 5. Тестирование ---
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nТочность на тестовых данных: {100 * correct / total:.2f}%")

# --- 6. График обучения ---
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs + 1), train_losses, 'b-o', linewidth=2)
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.title("Процесс обучения")
plt.grid(True, alpha=0.3)
plt.show()
```

### 4.3. Использование предобученной модели (Transfer Learning)

**Transfer Learning** — использование модели, обученной на большом датасете (ImageNet), для решения своей задачи:

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import urllib.request

# --- 1. Загрузка предобученной модели ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()  # Переводим в режим предсказания

# --- 2. Подготовка изображения ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Загружаем и обрабатываем изображение
img = Image.open("photo.jpg")
input_tensor = preprocess(img).unsqueeze(0)  # Добавляем batch dimension

# --- 3. Предсказание ---
with torch.no_grad():
    output = model(input_tensor)

# Получаем вероятности через Softmax
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# --- 4. Загрузка названий классов ImageNet ---
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(url, "imagenet_classes.txt")

with open("imagenet_classes.txt") as f:
    categories = [line.strip() for line in f.readlines()]

# Топ-5 предсказаний
top5_prob, top5_idx = torch.topk(probabilities, 5)
print("Топ-5 предсказаний:")
for i in range(5):
    print(f"  {categories[top5_idx[i]]:30s} — {top5_prob[i].item()*100:.2f}%")
```

### 4.4. Дообучение (Fine-tuning) для своих данных

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Структура папок для своих данных:
# dataset/
#   train/
#     cats/     ← изображения котов
#     dogs/     ← изображения собак
#   val/
#     cats/
#     dogs/

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),     # Аугментация: горизонтальное отражение
    transforms.RandomRotation(10),         # Аугментация: поворот ±10°
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = ImageFolder("dataset/train", transform=transform)
val_data = ImageFolder("dataset/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

print(f"Классы: {train_data.classes}")  # ['cats', 'dogs']

# --- Загрузка предобученной модели и замена последнего слоя ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Замораживаем все слои (не обучаем их)
for param in model.parameters():
    param.requires_grad = False

# Заменяем последний полносвязный слой
num_classes = len(train_data.classes)  # 2 класса
model.fc = nn.Linear(model.fc.in_features, num_classes)

# --- Обучение только последнего слоя ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Эпоха {epoch+1}/10, loss = {running_loss/len(train_loader):.4f}")

# Сохранение модели
torch.save(model.state_dict(), "my_classifier.pth")
```

---

## 5. Детекция объектов (Object Detection)

### 5.1. Что такое детекция?

**Классификация** отвечает на вопрос *«что на изображении?»*, а **детекция** — *«что и где?»*:

```
Классификация:              Детекция:
┌──────────────┐            ┌──────────────┐
│              │            │  ┌──────┐    │
│    🐱       │            │  │ 🐱  │    │
│              │            │  │ кот  │    │
│        🐕   │ → "кот"    │  └──────┘    │ → "кот" (x1,y1,x2,y2)
│              │            │      ┌─────┐ │   "собака" (x1,y1,x2,y2)
│              │            │      │ 🐕 │ │
└──────────────┘            │      │соба-│ │
                            │      │ка   │ │
                            │      └─────┘ │
                            └──────────────┘
```

### 5.2. Использование YOLOv5 для детекции объектов

```python
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. Загрузка предобученной модели YOLOv5 ---
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# --- 2. Детекция объектов на изображении ---
img = Image.open("photo.jpg")
results = model(img)

# --- 3. Вывод результатов ---
results.print()   # Текстовый результат

# Таблица с обнаруженными объектами
print(results.pandas().xyxy[0])
# Столбцы: xmin, ymin, xmax, ymax, confidence, class, name

# --- 4. Визуализация результатов ---
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img)

detections = results.pandas().xyxy[0]
colors = plt.cm.Set3(range(len(detections)))

for idx, det in detections.iterrows():
    x1, y1, x2, y2 = det['xmin'], det['ymin'], det['xmax'], det['ymax']
    conf = det['confidence']
    name = det['name']

    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor=colors[idx], facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(x1, y1 - 5, f"{name} {conf:.2f}",
            color='white', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.8))

ax.axis('off')
plt.title("Результат детекции YOLOv5")
plt.tight_layout()
plt.show()

# --- 5. Сохранение результата ---
results.save()  # Сохраняет в папку runs/detect/
```

### 5.3. Популярные архитектуры детекции

| Модель | Год | Скорость | Точность | Особенность |
|---|---|---|---|---|
| **R-CNN** | 2014 | Медленная | Высокая | Первый двухэтапный детектор |
| **YOLO** | 2016 | Быстрая | Средняя | Один проход (real-time) |
| **SSD** | 2016 | Быстрая | Средняя | Многомасштабная детекция |
| **YOLOv5** | 2020 | Очень быстрая | Высокая | PyTorch, удобный API |
| **YOLOv8** | 2023 | Очень быстрая | Очень высокая | Новейшая версия |

---

## 6. Сегментация изображений

### 6.1. Что такое сегментация?

**Сегментация** — это разметка каждого пикселя изображения с присвоением ему класса:

```
Вход:                     Сегментация:
┌───────────────┐         ┌───────────────┐
│   ☁️         │         │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  ▓ = небо
│               │         │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
│    🏠        │         │░░░░████░░░░░░░│  █ = здание
│               │         │░░░░████░░░░░░░│  ░ = трава
│🌳    🌳     │         │▒▒░░░░░░▒▒░░░░░│  ▒ = дерево
└───────────────┘         └───────────────┘
```

### 6.2. Типы сегментации

| Тип | Описание | Пример |
|---|---|---|
| **Семантическая** | Каждый пиксель получает класс | Все люди = один цвет |
| **Экземплярная** | Различает отдельные объекты | Человек 1 ≠ Человек 2 |
| **Паноптическая** | Семантическая + экземплярная | Полная разметка сцены |

### 6.3. Сегментация с помощью предобученной модели

```python
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Загрузка модели ---
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
model.eval()

# --- 2. Подготовка изображения ---
preprocess = weights.transforms()

img = Image.open("photo.jpg")
input_tensor = preprocess(img).unsqueeze(0)

# --- 3. Сегментация ---
with torch.no_grad():
    output = model(input_tensor)['out'][0]

# Для каждого пикселя берём класс с максимальной вероятностью
segmentation = output.argmax(0).numpy()

# --- 4. Визуализация ---
# Цветовая палитра для 21 класса PASCAL VOC
palette = np.array([
    [0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128],
    [128,0,128], [0,128,128], [128,128,128], [64,0,0], [192,0,0],
    [64,128,0], [192,128,0], [64,0,128], [192,0,128], [64,128,128],
    [192,128,128], [0,64,0], [128,64,0], [0,192,0], [128,192,0],
    [0,64,128]
], dtype=np.uint8)

# Названия классов
class_names = [
    'фон', 'самолёт', 'велосипед', 'птица', 'лодка', 'бутылка',
    'автобус', 'автомобиль', 'кот', 'стул', 'корова', 'стол',
    'собака', 'лошадь', 'мотоцикл', 'человек', 'растение',
    'овца', 'диван', 'поезд', 'монитор'
]

# Создаём цветную маску
color_mask = palette[segmentation]
color_mask_img = Image.fromarray(color_mask)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img)
axes[0].set_title("Оригинал")
axes[1].imshow(color_mask_img)
axes[1].set_title("Сегментация")
axes[2].imshow(img)
axes[2].imshow(color_mask_img, alpha=0.5)  # Наложение
axes[2].set_title("Наложение")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

# Вывод найденных классов
unique_classes = np.unique(segmentation)
print("Найденные объекты:")
for cls in unique_classes:
    pixels = (segmentation == cls).sum()
    pct = 100 * pixels / segmentation.size
    print(f"  {class_names[cls]:15s} — {pct:.1f}% пикселей")
```

---

## 7. Генерация и улучшение изображений

### 7.1. Суперразрешение (Super-Resolution)

Увеличение разрешения изображения с помощью нейронной сети:

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Загрузка предобученной модели ESRGAN
# pip install basicsr realesrgan
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --- 1. Настройка модели ---
model_net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)

upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth',  # Скачать веса заранее
    model=model_net,
    tile=0,
    half=False
)

# --- 2. Увеличение разрешения ---
import cv2
img = cv2.imread("low_res_photo.jpg")
output, _ = upsampler.enhance(img, outscale=4)

cv2.imwrite("high_res_photo.jpg", output)
print(f"Было: {img.shape[:2]} → Стало: {output.shape[:2]}")
```

### 7.2. Удаление шума (Denoising)

```python
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Простая модель автоэнкодера для удаления шума ---
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Энкодер: сжимает изображение
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # Декодер: восстанавливает
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = DenoisingAutoencoder()
print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

# Для тренировки:
# 1. Берём чистые изображения
# 2. Добавляем шум: noisy = clean + noise
# 3. Обучаем модель предсказывать чистые из зашумлённых
# 4. loss = MSE(model(noisy), clean)
```

### 7.3. Генеративно-состязательные сети (GAN)

GAN состоит из двух сетей, соревнующихся друг с другом:

```
                    ┌────────────────┐
  Случайный шум ──→ │   Генератор G  │ ──→ Fake Image
                    └────────────────┘         │
                                               ↓
                                        ┌──────────────┐
                    Реальное изображение │              │
                    ──────────────────→  │ Дискримина-  │ ──→ Real / Fake?
                                        │  тор D       │
                                        └──────────────┘

  G хочет обмануть D (создать реалистичное изображение)
  D хочет отличить настоящие изображения от поддельных
  В результате соревнования G учится генерировать всё лучше
```

---

## 8. Перенос стиля (Style Transfer)

### 8.1. Что такое перенос стиля?

Перенос стиля — это применение художественного стиля одного изображения к содержимому другого:

```
Контент (фотография)  +  Стиль (картина)  =  Результат

┌────────────┐         ┌────────────┐       ┌────────────┐
│    🏠      │    +    │ ~~~~~~~~~~~~│   =   │  ~~🏠~~   │
│   🌳🌳   │         │~~~~~~~~~~~~~│       │ ~~🌳🌳~~ │
│            │         │  Ван Гог    │       │  в стиле   │
└────────────┘         └────────────┘       │  Ван Гога  │
                                            └────────────┘
```

### 8.2. Реализация с PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Загрузка изображений ---
def load_image(path, max_size=400):
    img = Image.open(path).convert('RGB')
    # Масштабируем
    ratio = max_size / max(img.size)
    new_size = tuple(int(s * ratio) for s in img.size)
    img = img.resize(new_size, Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)

content_img = load_image("photo.jpg")
style_img = load_image("style.jpg")

# --- 2. Извлечение признаков из VGG19 ---
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# Замораживаем параметры
for param in vgg.parameters():
    param.requires_grad = False

def get_features(image, model):
    """Извлекаем карты признаков из нескольких слоёв VGG19."""
    layers = {
        '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
        '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """Матрица Грама для стиля."""
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t()) / (c * h * w)

# --- 3. Оптимизация ---
content_features = get_features(content_img, vgg)
style_features = get_features(style_img, vgg)

# Вычисляем матрицы Грама для стиля
style_grams = {layer: gram_matrix(style_features[layer])
               for layer in style_features}

# Начинаем с копии контентного изображения
target = content_img.clone().requires_grad_(True)

# Веса
content_weight = 1e4
style_weight = 1e6

optimizer = optim.Adam([target], lr=0.003)

# --- 4. Перенос стиля ---
steps = 300
for step in range(1, steps + 1):
    target_features = get_features(target, vgg)

    # Content loss (контент из слоя conv4_2)
    content_loss = torch.mean(
        (target_features['conv4_2'] - content_features['conv4_2']) ** 2
    )

    # Style loss (стиль из всех слоёв)
    style_loss = 0
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    for layer in style_layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        style_loss += torch.mean((target_gram - style_gram) ** 2)
    style_loss /= len(style_layers)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Шаг {step}/{steps}, "
              f"content_loss = {content_loss.item():.4f}, "
              f"style_loss = {style_loss.item():.6f}")

# --- 5. Результат ---
def tensor_to_image(tensor):
    img = tensor.clone().detach().cpu().squeeze(0)
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    img = img.clamp(0, 1)
    return transforms.ToPILImage()(img)

result = tensor_to_image(target)
result.save("stylized_result.jpg")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(tensor_to_image(content_img))
axes[0].set_title("Контент")
axes[1].imshow(tensor_to_image(style_img))
axes[1].set_title("Стиль")
axes[2].imshow(result)
axes[2].set_title("Результат")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

## 9. Контрольные вопросы

1. Чем нейросетевой подход к обработке изображений отличается от классического?
2. Что такое искусственный нейрон? Из каких компонентов он состоит?
3. Какую роль выполняет функция активации? Почему нельзя обойтись без неё?
4. Чем отличается ReLU от Sigmoid? Почему ReLU более популярна?
5. Что такое обратное распространение ошибки (backpropagation)?
6. Почему для обработки изображений используют свёрточные сети (CNN), а не полносвязные?
7. Какие слои входят в типичную CNN? Опишите роль каждого.
8. Что «видит» каждый слой CNN (от первого к последнему)?
9. Что такое Transfer Learning? Какие преимущества он даёт?
10. Чем классификация изображений отличается от детекции и сегментации?
11. Как работает YOLO для детекции объектов? Почему она быстрая?
12. Что такое GAN? Опишите роли генератора и дискриминатора.
13. Напишите программу, которая использует предобученную ResNet18 для классификации изображения.

---

## 10. Список литературы и ресурсов

### Документация

| Ресурс | Ссылка |
|---|---|
| PyTorch — основной фреймворк | <https://pytorch.org/docs/stable/> |
| TorchVision — модели и трансформации | <https://pytorch.org/vision/stable/> |
| Ultralytics YOLOv5 | <https://github.com/ultralytics/yolov5> |
| Ultralytics YOLOv8 | <https://github.com/ultralytics/ultralytics> |
| OpenCV DNN модуль | <https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html> |

### Книги

1. Гудфеллоу И., Бенжио Й., Курвилль А. — **«Глубокое обучение»** (Deep Learning) — фундаментальный учебник.
2. Николенко С., Кадурин А., Архангельская Е. — **«Глубокое обучение. Погружение в мир нейронных сетей»** — на русском языке.
3. François Chollet — **«Deep Learning with Python»** — практическое руководство с Keras.

### Датасеты для практики

| Датасет | Описание | Классов | Изображений |
|---|---|---|---|
| **MNIST** | Рукописные цифры 28×28 | 10 | 70 000 |
| **CIFAR-10** | Объекты 32×32 | 10 | 60 000 |
| **ImageNet** | Объекты 224×224 | 1 000 | 1.2 млн |
| **COCO** | Детекция + сегментация | 80 | 330 000 |
| **PASCAL VOC** | Детекция + сегментация | 20 | 11 530 |

### Онлайн-курсы

- Stanford CS231n: «Convolutional Neural Networks for Visual Recognition» — <https://cs231n.stanford.edu/>
- Fast.ai: «Practical Deep Learning for Coders» — <https://course.fast.ai/>
- Coursera: «Deep Learning Specialization» — Andrew Ng
