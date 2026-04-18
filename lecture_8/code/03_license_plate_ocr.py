import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

# 1. Инициализируем модели
print("Загрузка YOLOv8...")
model = YOLO('yolov8n.pt') # Обычная детекция (находит машины)

print("Загрузка EasyOCR (это может занять время первый раз)...")
# Мы указываем английский и русский языки. 'gpu=False' - если нет видеокарты NVIDIA.
reader = easyocr.Reader(['en', 'ru'], gpu=False) 

cap = cv2.VideoCapture(0)
print("Наведите камеру на фотографию машины с номером!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Ищем объекты с помощью YOLO
    # классы COCO: 2 = car (машина), 3 = motorcycle, 5 = bus, 7 = truck
    results = model(frame, classes=[2, 3, 5, 7], conf=0.5, verbose=False)
    
    # Копируем кадр для отрисовки
    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Получаем координаты рамки машины (x1, y1) - левый верхний угол, (x2, y2) - правый нижний
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Отрисуем синюю рамку вокруг машины
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 3. Вырезаем (Crop) область с машиной
            # Важно: y идет первым в numpy-массивах!
            car_roi = frame[y1:y2, x1:x2]
            
            # Пропускаем, если вырезка получилась пустой/ошибочной
            if car_roi.size == 0:
                continue

            # 4. Распознавание текста на вырезанной машине с помощью EasyOCR
            # Мы ищем любой текст (в надежде, что это номерной знак)
            ocr_results = reader.readtext(car_roi)
            
            for (bbox, text, prob) in ocr_results:
                # Фильтруем результаты с низкой уверенностью
                if prob > 0.3:
                    print(f"Обнаружен текст/номер: {text} (уверенность: {prob:.2f})")
                    # Отрисовываем распознанный текст НАД рамкой КАДРА (а не вырезки)
                    # Выводим зеленым цветом
                    cv2.putText(annotated_frame, text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Car Plate OCR', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
