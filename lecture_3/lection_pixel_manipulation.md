# Чтение и попиксельное изменение изображений

---

## Содержание

1. [Как изображение хранится в памяти](#1-как-изображение-хранится-в-памяти)
2. [Цветовые модели](#2-цветовые-модели)
3. [Создание изображений «с нуля» (NumPy)](#3-создание-изображений-с-нуля-numpy)
4. [Загрузка и сохранение изображений](#4-загрузка-и-сохранение-изображений)
5. [Попиксельный доступ: чтение и запись](#5-попиксельный-доступ-чтение-и-запись)
6. [Конвертация в оттенки серого (Grayscale)](#6-конвертация-в-оттенки-серого-grayscale)
7. [Бинаризация (чёрно-белое изображение)](#7-бинаризация-чёрно-белое-изображение)
8. [Негатив изображения](#8-негатив-изображения)
9. [Регулировка яркости и контраста](#9-регулировка-яркости-и-контраста)
10. [Работа с областями (ROI)](#10-работа-с-областями-roi)
11. [Попиксельные фильтры: размытие и резкость](#11-попиксельные-фильтры-размытие-и-резкость)
12. [Рисование примитивов на изображении](#12-рисование-примитивов-на-изображении)
13. [Контрольные вопросы](#14-контрольные-вопросы)
14. [Список литературы и ресурсов](#15-список-литературы-и-ресурсов)

---

## 1. Как изображение хранится в памяти

### 1.1. Растровое изображение — это матрица

Любое **растровое изображение** (bitmap) — это двумерная сетка (матрица) точек, называемых **пикселями** (pixel = **pic**ture + **el**ement).

```
┌─────┬─────┬─────┬─────┬─────┐
│ P00 │ P01 │ P02 │ P03 │ P04 │   ← строка 0
├─────┼─────┼─────┼─────┼─────┤
│ P10 │ P11 │ P12 │ P13 │ P14 │   ← строка 1
├─────┼─────┼─────┼─────┼─────┤
│ P20 │ P21 │ P22 │ P23 │ P24 │   ← строка 2
├─────┼─────┼─────┼─────┼─────┤
│ P30 │ P31 │ P32 │ P33 │ P34 │   ← строка 3
└─────┴─────┴─────┴─────┴─────┘
  ↑ столбец 0                ↑ столбец 4
```

**Ключевые характеристики:**

| Параметр | Описание |
|---|---|
| **Ширина (Width)** | Количество пикселей по горизонтали |
| **Высота (Height)** | Количество пикселей по вертикали |
| **Глубина цвета (Bit Depth)** | Количество бит на один канал (обычно 8 → значения 0–255) |
| **Количество каналов** | 1 (ЧБ), 3 (цветное), 4 (цветное с прозрачностью) |

### 1.2. Координатная система

В компьютерном представлении начало координат `(0, 0)` расположено в **левом верхнем** углу:

```
(0,0) ───────→ X (столбцы, ширина)
  │
  │
  ↓
  Y (строки, высота)
```

> **Важно:** В NumPy и OpenCV координаты записываются как `[y, x]`, то есть сначала **строка**, потом **столбец**. Это частый источник ошибок!

### 1.3. Объём памяти

Размер несжатого изображения в памяти вычисляется как:

```
Размер = Ширина × Высота × Каналы × (Глубина / 8) байт
```

**Примеры:**

| Изображение | Расчёт | Размер |
|---|---|---|
| ЧБ 100×100 | 100 × 100 × 1 × 1 | 10 КБ |
| RGB 100×100 | 100 × 100 × 3 × 1 | ≈ 3 МБ |
| RGBA 100×100 | 100 × 100 × 4 × 1 | ≈ 4 МБ |

---

## 2. Цветовые модели

### 2.1. Grayscale (оттенки серого)

Каждый пиксель хранит **одно** значение яркости от `0` (чёрный) до `255` (белый):

```
0 ──────── 128 ──────── 255
чёрный     серый        белый
```

### 2.2. RGB (Red, Green, Blue)

Каждый пиксель описывается тремя каналами: красный, зелёный, синий. Каждый канал принимает значения от `0` до `255`.

| Цвет | R | G | B |
|---|---|---|---|
| Чёрный | 0 | 0 | 0 |
| Белый | 255 | 255 | 255 |
| Красный | 255 | 0 | 0 |
| Зелёный | 0 | 255 | 0 |
| Синий | 0 | 0 | 255 |
| Жёлтый | 255 | 255 | 0 |
| Голубой | 0 | 255 | 255 |
| Пурпурный | 255 | 0 | 255 |
| Серый (50%) | 128 | 128 | 128 |

### 2.3. BGR (Blue, Green, Red) — формат OpenCV

**OpenCV** по умолчанию загружает изображения в формате **BGR** (каналы в обратном порядке). Это историческое наследие. При работе с OpenCV нужно помнить:

```
OpenCV (BGR):   пиксель = [B, G, R] = [синий, зелёный, красный]
Matplotlib/PIL: пиксель = [R, G, B] = [красный, зелёный, синий]
```

### 2.4. RGBA (с альфа-каналом)

Четвёртый канал **Alpha** (A) определяет **прозрачность** пикселя:

- `A = 0` — полностью прозрачный
- `A = 255` — полностью непрозрачный

Используется в формате **PNG** и при наложении слоёв.

### 2.5. HSV (Hue, Saturation, Value)

Альтернативная модель, удобная для определения цвета «на глаз»:

| Компонента | Описание | Диапазон (OpenCV) |
|---|---|---|
| **H — Тон** (Hue) | Оттенок цвета на цветовом круге | 0–179 |
| **S — Насыщенность** (Saturation) | Чистота цвета (0 = серый) | 0–255 |
| **V — Яркость** (Value) | Светлота (0 = чёрный) | 0–255 |

```
┌──────────────────────────────────────────────────┐
│  H:   0°=красный  60°=жёлтый  120°=зелёный      │
│       180°=голубой  240°=синий  300°=пурпурный    │
│  S:   0 (серый) ──────────────── 255 (чистый)    │
│  V:   0 (чёрный) ─────────────── 255 (яркий)    │
└──────────────────────────────────────────────────┘
```

---

## 3. Создание изображений «с нуля» (NumPy)

### 3.1. Чёрное изображение (все нули)

```python
import numpy as np
import matplotlib.pyplot as plt

# Создаём ЧБ-изображение 4×4 (матрица нулей)
# dtype=np.uint8 обязателен: пиксели — целые числа от 0 до 255
img_gray = np.zeros((4, 4), dtype=np.uint8)
print("Матрица:\n", img_gray)

# Визуализация
plt.figure(figsize=(3, 3))
plt.imshow(img_gray, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
plt.title("Чёрное изображение 4×4")
plt.axis('off')
plt.show()
```

**Вывод:**

```
Матрица:
 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]
```

### 3.2. Белое изображение

```python
img_white = np.full((4, 4), 255, dtype=np.uint8)
# Альтернатива: np.ones((4, 4), dtype=np.uint8) * 255
print("Матрица:\n", img_white)
```

### 3.3. Цветное изображение (RGB)

```python
# Цветное изображение — трёхмерная матрица: (высота, ширина, 3)
img_color = np.zeros((4, 4, 3), dtype=np.uint8)

# Закрасим первую строку красным
img_color[0, :] = [255, 0, 0]  # [R, G, B]

# Вторую строку — зелёным
img_color[1, :] = [0, 255, 0]

# Третью — синим
img_color[2, :] = [0, 0, 255]

# Четвёртую — жёлтым
img_color[3, :] = [255, 255, 0]

plt.figure(figsize=(3, 3))
plt.imshow(img_color, interpolation='nearest')
plt.title("Цветное изображение 4×4")
plt.axis('off')
plt.show()
```

### 3.4. Шахматная доска

```python
# Создаём шахматную доску 8×8
board = np.zeros((8, 8), dtype=np.uint8)

# Заполняем белые клетки
for y in range(8):
    for x in range(8):
        if (y + x) % 2 == 0:
            board[y, x] = 255

plt.figure(figsize=(4, 4))
plt.imshow(board, cmap='gray', interpolation='nearest')
plt.title("Шахматная доска 8×8")
plt.axis('off')
plt.show()
```

### 3.5. Градиентное изображение

```python
# Горизонтальный градиент: от чёрного к белому
gradient = np.zeros((100, 256), dtype=np.uint8)

for x in range(256):
    gradient[:, x] = x  # Каждый столбец — своя яркость

plt.figure(figsize=(8, 2))
plt.imshow(gradient, cmap='gray', interpolation='nearest')
plt.title("Горизонтальный градиент")
plt.axis('off')
plt.show()
```

---

## 4. Загрузка и сохранение изображений

### 4.1. С помощью Pillow (PIL)

```python
from PIL import Image
import numpy as np

# --- Загрузка ---
img_pil = Image.open("photo.jpg")
print(f"Формат: {img_pil.format}")        # JPEG
print(f"Размер: {img_pil.size}")           # (ширина, высота)
print(f"Режим:  {img_pil.mode}")           # RGB, L (ЧБ), RGBA и т.д.

# Преобразование PIL → NumPy
img_array = np.array(img_pil)
print(f"Массив: {img_array.shape}")        # (высота, ширина, каналы)

# --- Сохранение ---
# NumPy → PIL → файл
result = Image.fromarray(img_array)
result.save("output.png")

# Прямое сохранение с параметрами
img_pil.save("output.jpg", quality=85)
```

### 4.2. С помощью OpenCV

```python
import cv2

# --- Загрузка ---
img = cv2.imread("photo.jpg")              # Загрузка в BGR
img_gray = cv2.imread("photo.jpg", 0)      # Загрузка сразу в ЧБ
img_rgba = cv2.imread("photo.png", cv2.IMREAD_UNCHANGED)  # С альфа-каналом

print(f"Размерность: {img.shape}")          # (высота, ширина, каналы)
print(f"Тип данных:  {img.dtype}")          # uint8

# --- Конвертация BGR ↔ RGB ---
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Для matplotlib
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # Обратно для OpenCV

# --- Сохранение ---
cv2.imwrite("output.jpg", img)             # Параметр quality:
cv2.imwrite("output.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
cv2.imwrite("output.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
```

### 4.3. С помощью matplotlib (только для простых случаев)

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("photo.jpg")   # Загружает в RGB, float32 [0.0–1.0] или uint8
plt.imshow(img)
plt.show()

plt.imsave("output.png", img)
```

### 4.4. Сравнение библиотек

| Особенность | Pillow | OpenCV | matplotlib |
|---|---|---|---|
| Порядок каналов | RGB | **BGR** | RGB |
| Координаты | `(x, y)` через `.getpixel()` | `[y, x]` через NumPy | `[y, x]` |
| Скорость | Средняя | Высокая | Низкая |
| Обработка видео | ✗ | ✓ | ✗ |
| Фильтры и CV | Базовые | Полный набор | Только отображение |

---

## 5. Попиксельный доступ: чтение и запись

### 5.1. Чтение значения пикселя

```python
import numpy as np
from PIL import Image

img = np.array(Image.open("photo.jpg"))

# Координаты: [строка (y), столбец (x)]
y, x = 50, 100

# Для цветного изображения (RGB)
pixel = img[y, x]          # Массив [R, G, B]
r, g, b = img[y, x]        # Распаковка по каналам
print(f"Пиксель ({y}, {x}): R={r}, G={g}, B={b}")

# Для ЧБ-изображения
gray_img = np.array(Image.open("photo.jpg").convert('L'))
brightness = gray_img[y, x]    # Одно число 0–255
print(f"Яркость: {brightness}")
```

### 5.2. Запись значения пикселя

```python
# Одиночный пиксель
img[50, 100] = [255, 0, 0]       # Красный

# Горизонтальная линия (строка 0, все столбцы)
img[0, :] = [0, 255, 0]          # Зелёная линия сверху

# Вертикальная линия (все строки, столбец 0)
img[:, 0] = [0, 0, 255]          # Синяя линия слева

# Прямоугольная область
img[10:60, 20:80] = [255, 255, 0]  # Жёлтый прямоугольник

# Заполнение всего изображения
img[:, :] = [128, 128, 128]       # Полностью серое
```

### 5.3. Обход всех пикселей в цикле

```python
import numpy as np
from PIL import Image

img = np.array(Image.open("photo.jpg"))
height, width, channels = img.shape

# Медленный способ (для понимания)
for y in range(height):
    for x in range(width):
        r, g, b = img[y, x]
        # ... обработка каждого пикселя ...

# Быстрый способ (векторизация NumPy)
# Например, увеличить красный канал на 50:
img[:, :, 0] = np.clip(img[:, :, 0].astype(np.int16) + 50, 0, 255).astype(np.uint8)
```

> **Совет:** Всегда старайтесь использовать **векторизованные операции** NumPy вместо двойного цикла. На изображении 1920×1080 разница в скорости может быть в **100–1000 раз**.

### 5.4. Попиксельный доступ в Pillow

```python
from PIL import Image

img = Image.open("photo.jpg")

# Чтение пикселя
pixel = img.getpixel((100, 50))  # (x, y) — внимание на порядок!
print(f"Пиксель: {pixel}")       # (R, G, B)

# Запись пикселя
img.putpixel((100, 50), (255, 0, 0))

# Массовый доступ через load()
pixels = img.load()
for y in range(img.height):
    for x in range(img.width):
        r, g, b = pixels[x, y]   # (x, y) — как в Pillow
        pixels[x, y] = (r, g, b)

img.save("modified.jpg")
```

> **Внимание:** В Pillow координаты передаются как `(x, y)`, а в NumPy/OpenCV — как `[y, x]`!

---

## 6. Конвертация в оттенки серого (Grayscale)

### 6.1. Зачем нужен Grayscale?

- Уменьшение объёма данных в 3 раза (1 канал вместо 3)
- Упрощение алгоритмов обработки
- Подготовка к бинаризации, детекции границ и т.д.
- Многие алгоритмы CV работают только с ЧБ

### 6.2. Формула конвертации

Человеческий глаз воспринимает цвета неравномерно: зелёный — ярче всего, синий — тусклее. Поэтому используется **взвешенная формула (стандарт ITU-R BT.601)**:

```
Gray = 0.299 × R + 0.587 × G + 0.114 × B
```

| Канал | Вес | Почему? |
|---|---|---|
| R (красный) | 0.299 | Средняя чувствительность глаза |
| G (зелёный) | 0.587 | Наибольшая чувствительность |
| B (синий) | 0.114 | Наименьшая чувствительность |

### 6.3. Реализация вручную (попиксельно)

```python
import numpy as np
from PIL import Image

img = np.array(Image.open("photo.jpg"))
height, width, _ = img.shape

# Создаём пустую ЧБ-матрицу
gray = np.zeros((height, width), dtype=np.uint8)

# Попиксельная конвертация
for y in range(height):
    for x in range(width):
        r, g, b = img[y, x]
        gray[y, x] = int(0.299 * r + 0.587 * g + 0.114 * b)

# Визуализация
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title("Оригинал (RGB)")
axes[0].axis('off')
axes[1].imshow(gray, cmap='gray')
axes[1].set_title("Оттенки серого")
axes[1].axis('off')
plt.show()
```

### 6.4. Быстрая реализация (NumPy)

```python
# Одна строчка вместо двойного цикла
gray_fast = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
```

### 6.5. С помощью библиотек

```python
# Pillow
from PIL import Image
gray_pil = Image.open("photo.jpg").convert('L')
gray_pil.save("gray_pillow.jpg")

# OpenCV
import cv2
img_bgr = cv2.imread("photo.jpg")
gray_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_opencv.jpg", gray_cv)
```

### 6.6. Простое среднее vs. взвешенное

Для сравнения — как выглядит «наивное» среднее:

```python
# Наивное среднее (менее точное для человеческого восприятия)
gray_avg = img.mean(axis=2).astype(np.uint8)

# Взвешенное (стандарт BT.601 — более естественное)
gray_weighted = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8)
```

---

## 7. Бинаризация (чёрно-белое изображение)

### 7.1. Что такое бинаризация?

**Бинаризация** — превращение изображения в строго чёрно-белое, где каждый пиксель принимает только два значения: `0` (чёрный) или `255` (белый).

```
Серое:    [0 ... 127 ... 128 ... 255]
                  ↓ порог = 128
Бинарное: [0  0  0  ... 0 | 255  255 ... 255]
```

### 7.2. Применение

- Распознавание текста (OCR)
- Обнаружение объектов (контуры)
- Сегментация изображений
- Сканирование документов

### 7.3. Простая пороговая бинаризация

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Загружаем и конвертируем в ЧБ
img = np.array(Image.open("photo.jpg").convert('L'))

threshold = 128  # Пороговое значение

# Попиксельная бинаризация
height, width = img.shape
binary = np.zeros_like(img)

for y in range(height):
    for x in range(width):
        if img[y, x] >= threshold:
            binary[y, x] = 255
        else:
            binary[y, x] = 0

# Быстрая версия (NumPy)
binary_fast = np.where(img >= threshold, 255, 0).astype(np.uint8)

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Оттенки серого")
axes[1].imshow(binary, cmap='gray')
axes[1].set_title(f"Бинаризация (порог={threshold})")
axes[2].imshow(binary_fast, cmap='gray')
axes[2].set_title("Бинаризация (NumPy)")
for ax in axes:
    ax.axis('off')
plt.show()
```

### 7.4. Выбор порога: влияние на результат

```python
thresholds = [64, 128, 192]

fig, axes = plt.subplots(1, len(thresholds) + 1, figsize=(16, 4))
axes[0].imshow(img, cmap='gray')
axes[0].set_title("Оригинал")
axes[0].axis('off')

for i, t in enumerate(thresholds):
    binary_t = np.where(img >= t, 255, 0).astype(np.uint8)
    axes[i+1].imshow(binary_t, cmap='gray')
    axes[i+1].set_title(f"Порог = {t}")
    axes[i+1].axis('off')

plt.tight_layout()
plt.show()
```

### 7.5. Метод Оцу (автоматический выбор порога)

Метод Оцу **автоматически** находит оптимальный порог, минимизируя внутриклассовую дисперсию:

```python
import cv2

img_gray = cv2.imread("photo.jpg", 0)

# Метод Оцу
otsu_thresh, binary_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Порог Оцу: {otsu_thresh}")

# Обычная бинаризация для сравнения
_, binary_manual = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
```

### 7.6. Адаптивная бинаризация

При неравномерном освещении фиксированный порог работает плохо. **Адаптивная бинаризация** вычисляет порог для каждой локальной области:

```python
import cv2

img_gray = cv2.imread("document.jpg", 0)

# Адаптивная бинаризация (среднее значение окрестности)
adaptive_mean = cv2.adaptiveThreshold(
    img_gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,     # Метод: среднее
    cv2.THRESH_BINARY,
    blockSize=11,                    # Размер окрестности (нечётное число)
    C=2                              # Константа, вычитаемая из среднего
)

# Адаптивная бинаризация (метод Гаусса)
adaptive_gauss = cv2.adaptiveThreshold(
    img_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Метод: взвешенное Гауссово
    cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)
```

---

## 8. Негатив изображения

### 8.1. Принцип

Негатив — это инвертирование всех значений пикселей:

```
новое_значение = 255 - старое_значение
```

### 8.2. Реализация

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = np.array(Image.open("photo.jpg"))

# Попиксельно (для понимания)
negative = np.zeros_like(img)
height, width, channels = img.shape
for y in range(height):
    for x in range(width):
        for c in range(channels):
            negative[y, x, c] = 255 - img[y, x, c]

# NumPy (быстро)
negative_fast = 255 - img

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title("Оригинал")
axes[1].imshow(negative_fast)
axes[1].set_title("Негатив")
for ax in axes:
    ax.axis('off')
plt.show()
```

---

## 9. Регулировка яркости и контраста

### 9.1. Яркость

Яркость изменяется простым **сложением** константы с каждым пикселем:

```
новый_пиксель = старый_пиксель + β
```

```python
import numpy as np

def adjust_brightness(img, beta):
    """Изменяет яркость на значение beta (может быть отрицательным)."""
    result = img.astype(np.int16) + beta   # int16 чтобы не потерять переполнение
    return np.clip(result, 0, 255).astype(np.uint8)

# Увеличить яркость на 50
bright = adjust_brightness(img, 50)

# Уменьшить яркость на 50
dark = adjust_brightness(img, -50)
```

### 9.2. Контраст

Контраст изменяется **умножением** каждого пикселя на коэффициент:

```
новый_пиксель = α × старый_пиксель
```

```python
def adjust_contrast(img, alpha):
    """Изменяет контраст. alpha > 1 — увеличение, alpha < 1 — уменьшение."""
    result = img.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)

# Увеличить контраст в 1.5 раза
high_contrast = adjust_contrast(img, 1.5)

# Уменьшить контраст
low_contrast = adjust_contrast(img, 0.5)
```

### 9.3. Яркость + контраст одновременно

```python
def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """
    Общая формула: новый = alpha * старый + beta
    alpha — контраст (1.0 = без изменений)
    beta  — яркость (0 = без изменений)
    """
    result = img.astype(np.float32) * alpha + beta
    return np.clip(result, 0, 255).astype(np.uint8)

# Пример: повысить контраст на 30% и яркость на 20
adjusted = adjust_brightness_contrast(img, alpha=1.3, beta=20)
```

> **`np.clip()`** — ограничивает значения диапазоном [0, 255], предотвращая переполнение.

---

## 10. Работа с областями (ROI)

### 10.1. Вырезание фрагмента (Crop)

```python
import numpy as np
from PIL import Image

img = np.array(Image.open("photo.jpg"))

# Вырезаем прямоугольник: строки 100–200, столбцы 150–300
# Формат: img[y_start:y_end, x_start:x_end]
crop = img[100:200, 150:300]

print(f"Оригинал: {img.shape}")    # (H, W, 3)
print(f"Фрагмент: {crop.shape}")   # (100, 150, 3)

# Сохраняем фрагмент
Image.fromarray(crop).save("crop.jpg")
```

### 10.2. Вставка фрагмента

```python
# Вставим вырезанный фрагмент в другое место
img_copy = img.copy()
h, w, _ = crop.shape
img_copy[0:h, 0:w] = crop  # Вставляем в левый верхний угол
```

### 10.3. Замена цвета в области

```python
# Сделаем красным квадрат 50×50 в центре
cy, cx = img.shape[0] // 2, img.shape[1] // 2
img_copy = img.copy()
img_copy[cy-25:cy+25, cx-25:cx+25] = [255, 0, 0]
```

### 10.4. Маскирование (выделение по цвету)

```python
import cv2
import numpy as np

img = cv2.imread("photo.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Диапазон красного цвета в HSV
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# Создаём маску: белое — где есть красный, чёрное — где нет
mask = cv2.inRange(hsv, lower_red, upper_red)

# Применяем маску
result = cv2.bitwise_and(img, img, mask=mask)
```

---

## 11. Попиксельные фильтры: размытие и резкость

### 11.1. Свёртка (Convolution)

Большинство фильтров работают через **свёртку** — «скольжение» маленькой матрицы (**ядра**, kernel) по изображению:

```
Ядро 3×3:                  Изображение:
┌───┬───┬───┐              X — текущий пиксель
│ a │ b │ c │              Соседи участвуют в вычислении
├───┼───┼───┤
│ d │ e │ f │    ─────→    Новый_X = a×P1 + b×P2 + c×P3 +
├───┼───┼───┤                        d×P4 + e×P5 + f×P6 +
│ g │ h │ i │                        g×P7 + h×P8 + i×P9
└───┴───┴───┘
```

### 11.2. Размытие (Blur) — «среднее по соседям»

```python
import numpy as np
from PIL import Image

img = np.array(Image.open("photo.jpg").convert('L'), dtype=np.float32)
height, width = img.shape
blurred = np.zeros_like(img)

# Простое Box Blur 3×3 (среднее значение 9 соседних пикселей)
for y in range(1, height - 1):
    for x in range(1, width - 1):
        total = 0.0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                total += img[y + dy, x + dx]
        blurred[y, x] = total / 9.0

blurred = np.clip(blurred, 0, 255).astype(np.uint8)
```

**С помощью OpenCV (быстро):**

```python
import cv2

img = cv2.imread("photo.jpg")

# Box Blur
blur_box = cv2.blur(img, (5, 5))

# Гауссово размытие (более естественное)
blur_gauss = cv2.GaussianBlur(img, (5, 5), 0)

# Медианное размытие (хорошо убирает «шум соль-перец»)
blur_median = cv2.medianBlur(img, 5)
```

### 11.3. Резкость (Sharpening)

Ядро для повышения резкости усиливает разницу между пикселем и его соседями:

```python
import cv2
import numpy as np

img = cv2.imread("photo.jpg")

# Ядро для резкости
kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

sharpened = cv2.filter2D(img, -1, kernel_sharpen)
```

### 11.4. Обнаружение границ (Edge Detection)

```python
import cv2

img_gray = cv2.imread("photo.jpg", 0)

# Оператор Собеля (горизонтальные и вертикальные границы)
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # по X
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # по Y

# Оператор Кэнни (самый популярный)
edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)
```

### 11.5. Ядра фильтров: справочная таблица

| Фильтр | Ядро 3×3 | Эффект |
|---|---|---|
| **Box Blur** | `[[1,1,1],[1,1,1],[1,1,1]] / 9` | Равномерное размытие |
| **Резкость** | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | Повышение чёткости |
| **Собель X** | `[[-1,0,1],[-2,0,2],[-1,0,1]]` | Вертикальные границы |
| **Собель Y** | `[[-1,-2,-1],[0,0,0],[1,2,1]]` | Горизонтальные границы |
| **Лаплас** | `[[0,1,0],[1,-4,1],[0,1,0]]` | Обнаружение границ |
| **Тиснение** | `[[-2,-1,0],[-1,1,1],[0,1,2]]` | Эффект рельефа |

---

## 12. Рисование примитивов на изображении

### 12.1. С помощью OpenCV

```python
import cv2
import numpy as np

# Создаём чёрное полотно 400×600
canvas = np.zeros((400, 600, 3), dtype=np.uint8)

# Линия
cv2.line(canvas, (50, 50), (550, 50), (255, 255, 255), 2)
# Аргументы: изображение, точка1, точка2, цвет (BGR), толщина

# Прямоугольник
cv2.rectangle(canvas, (50, 80), (250, 200), (0, 255, 0), 2)
# Заполненный прямоугольник: толщина = -1
cv2.rectangle(canvas, (300, 80), (500, 200), (0, 0, 255), -1)

# Круг
cv2.circle(canvas, (150, 300), 50, (255, 0, 0), 2)
# Аргументы: изображение, центр, радиус, цвет, толщина

# Текст
cv2.putText(canvas, "Hello, OpenCV!", (50, 380),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imwrite("canvas.png", canvas)
```

### 12.2. С помощью Pillow

```python
from PIL import Image, ImageDraw, ImageFont

# Создаём белое полотно
canvas = Image.new('RGB', (600, 400), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

# Прямоугольник
draw.rectangle([50, 50, 200, 150], outline='red', width=2)

# Заполненный круг (эллипс)
draw.ellipse([250, 50, 400, 150], fill='blue')

# Линия
draw.line([(50, 200), (550, 200)], fill='green', width=3)

# Текст
draw.text((50, 250), "Привет, Pillow!", fill='black')

canvas.save("pillow_canvas.png")
```

## 13. Контрольные вопросы

1. Что такое пиксель и какие данные он хранит?
2. Чем отличается цветовая модель RGB от BGR? Почему OpenCV использует BGR?
3. Как расположена точка (0, 0) в координатной системе изображения?
4. Какой порядок координат используется в NumPy и Pillow при обращении к пикселю?
5. Что произойдёт, если записать в пиксель значение 300? Почему используется `np.clip()`?
6. Запишите формулу конвертации RGB → Grayscale по стандарту BT.601. Почему вес зелёного канала наибольший?
7. Что такое бинаризация? Приведите примеры практического применения.
8. Чем отличается метод Оцу от простой пороговой бинаризации?
9. Когда следует использовать адаптивную бинаризацию вместо глобальной?
10. Как работает свёртка (convolution)? Нарисуйте процесс применения ядра 3×3.
11. Чем отличается Гауссово размытие от Box Blur?
12. Как получить негатив изображения? Запишите формулу.
13. Каким образом можно изменить яркость и контраст изображения? Запишите общую формулу.
14. Создайте программу, которая загружает цветное изображение и создаёт из него 4 варианта: оригинал, оттенки серого, бинарное (порог 128) и негатив.

---

## 14. Список литературы и ресурсов

### Документация

| Ресурс | Ссылка |
|---|---|
| NumPy — работа с массивами | <https://numpy.org/doc/stable/> |
| Pillow (PIL Fork) | <https://pillow.readthedocs.io/> |
| OpenCV-Python | <https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html> |
| Matplotlib — визуализация | <https://matplotlib.org/stable/> |

### Книги

1. Гонсалес Р., Вудс Р. — **«Цифровая обработка изображений»** — классический учебник по теории.
2. Шапиро Л., Стокман Дж. — **«Компьютерное зрение»** — фундаментальный курс.
3. Adrian Rosebrock — **«Practical Python and OpenCV»** — практическое руководство.

### Стандарты

- **ITU-R BT.601** — стандарт кодирования видеосигнала (формула Grayscale)
- **ITU-R BT.709** — стандарт для HDTV (альтернативные коэффициенты: 0.2126, 0.7152, 0.0722)

### Онлайн-курсы

- OpenCV University — <https://opencv.org/university/>
- Coursera: «Image and Video Processing» — Duke University
