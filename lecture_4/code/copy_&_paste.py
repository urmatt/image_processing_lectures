import numpy as np
from PIL import Image
from pathlib import Path

cwd = Path(__file__).parent.parent

img = np.array(Image.open(cwd / "images" / "dream_of.jpg"))

# Вырезаем прямоугольник: строки 100–200, столбцы 150–300
# Формат: img[y_start:y_end, x_start:x_end]
crop = img[380:550, 580:800]

print(f"Оригинал: {img.shape}")    # (H, W, 3)
print(f"Фрагмент: {crop.shape}")   # (100, 150, 3)

# Вставим вырезанный фрагмент в другое место
img_copy = img.copy()
h, w, _ = crop.shape
img_copy[0:h, 0:w] = crop  # Вставляем в левый верхний угол

_path = Path(cwd / "images" / "dream_of_pasted.jpg")
_path.unlink(missing_ok=True)

# Сохраняем фрагмент
Image.fromarray(img_copy).save(_path)