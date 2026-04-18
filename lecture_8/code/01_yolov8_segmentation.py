import cv2
from ultralytics import YOLO

# Загружаем модель сегментации (обратите внимание на суффикс -seg)
model = YOLO('yolov8n-seg.pt')

# Открываем веб-камеру
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Запускаем инференс на кадре
    # Возвращается список результатов, берем первый (так как кадр один)
    results = model(frame)
    
    # YOLO может сама отрисовать результаты (маски + рамки)
    annotated_frame = results[0].plot()

    # Показываем результат
    cv2.imshow('YOLOv8 Instance Segmentation', annotated_frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
