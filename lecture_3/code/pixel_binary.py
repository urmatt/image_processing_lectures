"""
Лекция 3: Бинаризация изображения (чёрно-белое)

Демонстрирует:
  1. Простую пороговую бинаризацию с разными значениями порога
  2. Негатив изображения
  3. Изменение яркости и контраста
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

cwd = Path(__file__).parent.parent

# --- Загрузка и конвертация в ЧБ ---
image_path = cwd / "images/photo_1.jpg"
print(image_path)
img_rgb = np.array(Image.open(image_path))
img_gray = np.array(Image.open(image_path).convert('L'))

# --- 1. Бинаризация с разными порогами ---
thresholds = [64, 128, 192]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title("Оттенки серого")
axes[0].axis('off')

for i, t in enumerate(thresholds):
    binary = np.where(img_gray >= t, 255, 0).astype(np.uint8)
    axes[i + 1].imshow(binary, cmap='gray')
    axes[i + 1].set_title(f"Порог = {t}")
    axes[i + 1].axis('off')

plt.suptitle("Бинаризация с разными порогами", fontsize=14)
plt.tight_layout()
plt.show()

# --- 2. Негатив ---
negative = 255 - img_rgb

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_rgb)
axes[0].set_title("Оригинал")
axes[1].imshow(negative)
axes[1].set_title("Негатив")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

# --- 3. Яркость ---
def adjust_brightness(img, beta):
    result = img.astype(np.int16) + beta
    return np.clip(result, 0, 255).astype(np.uint8)

bright = adjust_brightness(img_rgb, 60)
dark = adjust_brightness(img_rgb, -60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(dark)
axes[0].set_title("Темнее (β = -60)")
axes[1].imshow(img_rgb)
axes[1].set_title("Оригинал")
axes[2].imshow(bright)
axes[2].set_title("Ярче (β = +60)")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

# --- 4. Контраст ---
def adjust_contrast(img, alpha):
    result = img.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)

low = adjust_contrast(img_rgb, 0.5)
high = adjust_contrast(img_rgb, 1.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(low)
axes[0].set_title("Контраст × 0.5")
axes[1].imshow(img_rgb)
axes[1].set_title("Оригинал")
axes[2].imshow(high)
axes[2].set_title("Контраст × 1.5")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
