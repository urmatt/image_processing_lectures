import numpy as np
from PIL import Image
from pathlib import Path

cwd = Path(__file__).parent.parent

img = np.array(Image.open(cwd / "images" / "dream_of.jpg"))

# Вырезаем прямоугольник: строки 100–200, столбцы 150–300
# Формат: img[y_start:y_end, x_start:x_end]
crop = img[350:600, 550:850]

print(f"Оригинал: {img.shape}")    # (H, W, 3)
print(f"Фрагмент: {crop.shape}")   # (100, 150, 3)

_path = Path(cwd / "images" / "dream_of_cropped.jpg")
_path.unlink(missing_ok=True)

# Сохраняем фрагмент
Image.fromarray(crop).save(_path)