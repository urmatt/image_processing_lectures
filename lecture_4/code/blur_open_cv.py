import cv2
from pathlib import Path
from PIL import Image

cwd = Path(__file__).parent.parent

img = cv2.imread(cwd / "images" / "best_view.jpg")

blur_box_size = 11

# Box Blur
blur_box = cv2.blur(img, (blur_box_size, blur_box_size))

# Гауссово размытие (более естественное)
blur_gauss = cv2.GaussianBlur(img, (blur_box_size, blur_box_size), 0)

# Медианное размытие (хорошо убирает «шум соль-перец»)
# blur_median = cv2.medianBlur(img, blur_box_size)

_path = Path(cwd / "images" / "best_view_blurred.jpg")
_path.unlink(missing_ok=True)

blur_box_rgb = cv2.cvtColor(blur_box, cv2.COLOR_BGR2RGB)
Image.fromarray(blur_box_rgb).save(_path)
