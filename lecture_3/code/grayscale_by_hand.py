from pathlib import Path

import numpy as np
from PIL import Image

cwd = Path(__file__).parent.parent

# --- Загрузка и конвертация в ЧБ ---
image_path = cwd / "images/photo_1.jpg"

img = np.array(Image.open(image_path))
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