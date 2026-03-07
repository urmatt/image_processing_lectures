from pathlib import Path

import numpy as np
from PIL import Image

cwd = Path(__file__).parent.parent

# --- Загрузка и конвертация в ЧБ ---
image_path = cwd / "images/photo_1.jpg"

img = np.array(Image.open(image_path))

# Координаты: [строка (y), столбец (x)]
y, x = 407, 611

# Для цветного изображения (RGB)
pixel = img[y, x]          # Массив [R, G, B]
r, g, b = img[y, x]        # Распаковка по каналам
height, width, _ = img.shape
print(f"Загружено: {width}×{height}, каналов: {img.shape[2]}")

print(f"Пиксель ({y}, {x}): R={r}, G={g}, B={b}")

# Для ЧБ-изображения
# gray_img = np.array(Image.open("photo.jpg").convert('L'))
# brightness = gray_img[y, x]    # Одно число 0–255
# print(f"Яркость: {brightness}")