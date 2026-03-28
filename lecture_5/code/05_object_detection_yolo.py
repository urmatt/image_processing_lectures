"""
Лекция 5 — Пример 5: Детекция объектов с помощью YOLO (Ultralytics).
Находит объекты на изображении, рисует рамки и подписи.

Установка: pip install ultralytics matplotlib pillow
"""
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import urllib.request
from pathlib import Path

cwd = Path(__file__).parent.parent

# ============================================================
# 1. Загрузка модели YOLOv8
# ============================================================
print("Загрузка модели YOLOv8s...")
model = YOLO("yolov8s.pt")  # Автоматически скачает веса
print("Модель загружена!")

# ============================================================
# 2. Подготовка изображения
# ============================================================
image_path = cwd / "images" / "sleeping_on.jpg"
if not os.path.exists(image_path):
    print(f"\nФайл '{image_path}' не найден. Скачиваем тестовое изображение...")
    os.makedirs(image_path.parent, exist_ok=True)
    url = "https://ultralytics.com/images/zidane.jpg"
    urllib.request.urlretrieve(url, image_path)
    print(f"Сохранено: {image_path}")

img = Image.open(image_path)
print(f"\nИзображение: {img.size[0]}×{img.size[1]}")

# ============================================================
# 3. Детекция объектов
# ============================================================
print("\nДетекция объектов...")
results = model(img)
result = results[0]

# Получаем данные обнаружений
boxes = result.boxes
names = result.names

print(f"\nОбнаружено объектов: {len(boxes)}")
for box in boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    name = names[cls_id]
    print(f"  {name:20s} — {conf:.0%}")

# ============================================================
# 4. Визуализация
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Оригинал
ax1.imshow(img)
ax1.set_title("Оригинал", fontsize=14)
ax1.axis('off')

# С детекциями
ax2.imshow(img)
colors = plt.cm.Set2(range(len(boxes)))

for idx, box in enumerate(boxes):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    name = names[cls_id]

    # Рамка
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2.5, edgecolor=colors[idx], facecolor='none'
    )
    ax2.add_patch(rect)

    # Подпись
    ax2.text(
        x1, y1 - 5,
        f"{name} {conf:.0%}",
        color='white', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor=colors[idx], alpha=0.85, edgecolor='none')
    )

ax2.set_title(f"Детекция YOLO ({len(boxes)} объектов)", fontsize=14)
ax2.axis('off')

plt.tight_layout()
output_path = cwd / "images" / "detection_result.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nСохранено: {output_path}")
