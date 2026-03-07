"""
Лекция 3: Конвертация изображения в оттенки серого (Grayscale)

Демонстрирует:
  1. Ручную попиксельную конвертацию RGB → Grayscale по формуле BT.601
  2. Быструю конвертацию через NumPy
  3. Конвертацию через Pillow
  4. Сравнение: наивное среднее vs. взвешенное
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# --- Загрузка ---
cwd = Path(__file__).parent.parent
image_path = cwd / "images/photo_1.jpg"
img = np.array(Image.open(image_path))
height, width, _ = img.shape
print(f"Загружено: {width}×{height}, каналов: {img.shape[2]}")

# --- 1. Попиксельная конвертация (BT.601) ---
gray_manual = np.zeros((height, width), dtype=np.uint8)
for y in range(height):
    for x in range(width):
        r, g, b = img[y, x]
        gray_manual[y, x] = int(0.299 * r + 0.587 * g + 0.114 * b)

# --- 2. Быстрая конвертация (NumPy) ---
gray_fast = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)

# --- 3. Наивное среднее ---
gray_avg = img.mean(axis=2).astype(np.uint8)

# --- 4. Pillow ---
gray_pil = np.array(Image.open(image_path).convert('L'))

# --- Визуализация ---
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(img)
axes[0, 0].set_title("Оригинал (RGB)")

axes[0, 1].imshow(gray_manual, cmap='gray')
axes[0, 1].set_title("Ручное (BT.601)")

axes[1, 0].imshow(gray_avg, cmap='gray')
axes[1, 0].set_title("Наивное среднее")

axes[1, 1].imshow(gray_pil, cmap='gray')
axes[1, 1].set_title("Pillow .convert('L')")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
