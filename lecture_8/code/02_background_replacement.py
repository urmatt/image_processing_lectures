import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 1. Загружаем модель
model = YOLO('yolov8n-seg.pt')

# 2. Загружаем фон (замените путь на свою картинку)
# Если картинки нет, скрипт вылетит с ошибкой. Создадим пустой зеленый фон (хромакей) по умолчанию
bg_image = np.zeros((480, 640, 3), dtype=np.uint8)
bg_image[:] = (0, 255, 0) # BGR формат (Green)

cap = cv2.VideoCapture(0)

# Получим ширину и высоту кадра камеры (чтобы подогнать фон)
ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    bg_image = cv2.resize(bg_image, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ищем объекты
    # classes=0 означает, что мы ищем ТОЛЬКО людей (класс 0 в датасете COCO)
    results = model(frame, classes=0, conf=0.5, verbose=False)
    
    result = results[0]
    
    # 3. Достаем маску
    # Создаем пустую общую маску такого же размера, как кадр камеры (только 1 канал - черно-белый)
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is not None:
        # masks.data содержит тензоры масок для каждого найденного человека
        # Это тензор PyTorch, переводим его в numpy массив и на процессор (cpu)
        masks = result.masks.data.cpu().numpy()
        
        for mask in masks:
             # YOLO возвращает маску небольшого размера (например, 160x160),
             # нам нужно масштабировать ее обратно под размер оригинального кадра (w, h)
             mask_resized = cv2.resize(mask, (w, h))
             
             # Объединяем маски, если людей несколько (побитовое ИЛИ)
             # Все, что больше 0.5 считаем объектом (белым)
             binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
             combined_mask = cv2.bitwise_or(combined_mask, binary_mask)

    # 4. Размываем края маски (Gaussian Blur), чтобы вырезание было мягким, без резких пикселей
    # Размер ядра (21, 21) управляет силой размытия. Должен быть нечетным!
    blurred_mask = cv2.GaussianBlur(combined_mask, (21, 21), 0)
    
    # Превращаем маску из [0..255] в [0.0..1.0] для математики
    alpha = blurred_mask.astype(float) / 255.0
    
    # Маска имеет 1 канал (ч/б), а картинка 3 канала (BGR). 
    # Дублируем маску на все 3 канала [H, W, 1] -> [H, W, 3]
    alpha_3d = np.dstack([alpha, alpha, alpha])
    
    # 5. Собираем финальное изображение
    # foreground - оставляем человека из оригинального кадра
    foreground = cv2.multiply(alpha_3d, frame.astype(float))
    
    # background - оставляем фон из bg_image, инвертируя маску (1.0 - alpha)
    background = cv2.multiply(1.0 - alpha_3d, bg_image.astype(float))
    
    # Складываем вместе и переводим обратно в формат opencv (uint8)
    final_output = cv2.add(foreground, background).astype(np.uint8)

    # Показываем результат
    cv2.imshow('Background Replacement', final_output)
    
    # Для отладки можем так же показать саму маску
    # cv2.imshow('Alpha Mask', combined_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
