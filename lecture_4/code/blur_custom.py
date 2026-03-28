import numpy as np
from PIL import Image
from pathlib import Path

cwd = Path(__file__).parent.parent

img = np.array(Image.open(cwd / "images" / "best_view.jpg"), dtype=np.float32)
height, width, channels = img.shape
blurred = np.zeros_like(img)

print("H: {}, W: {}, CH: {}".format(height, width, channels))

blur_size = 5  # Размер ядра (должен быть нечётным)
radius = blur_size // 2
kernel_area = blur_size * blur_size

# Простое Box Blur (среднее значение соседних пикселей)
for y in range(0, height - radius):
    for x in range(0, width - radius):
        for c in range(channels):
            total = 0.0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    total += img[y + dy, x + dx, c]
            blurred[y, x, c] = total / kernel_area
# 392 040 000
# 547 560 000 
# 3 113 640 000
_path = Path(cwd / "images" / "best_view_blurred_custom.jpg")
_path.unlink(missing_ok=True)

blurred = np.clip(blurred, 0, 255).astype(np.uint8)
Image.fromarray(blurred).save(_path)