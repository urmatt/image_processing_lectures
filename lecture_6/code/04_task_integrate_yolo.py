from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def main():

    # ============================================================
    # 1. Загрузка модели YOLOv8
    # ============================================================
    print("Загрузка модели YOLOv8s...")
    model = YOLO("yolov8m.pt")  # Автоматически скачает веса
    print("Модель загружена!")

    # Инициализация захвата видео (0 - первая доступная камера)
    cap = cv2.VideoCapture(0)

    # Проверка, что камера успешно открылась
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return

    print("Камера успешно запущена. Нажмите 'q' для выхода.")

    # Бесконечный цикл покадрового чтения
    while True:
        # read() возвращает флаг успешности и сам кадр
        ret, frame = cap.read()

        # Если кадр не получен, значит видео закончилось или произошел сбой
        if not ret:
            print("Не удалось получить кадр")
            break

        # Отразим кадр по горизонтали, чтобы он выглядел как в зеркале (опционально)
        frame = cv2.flip(frame, 1)

        # TODO: распознать обекты через YOLO
        # ============================================================
        # 3. Детекция объектов
        # ============================================================
        print("\nДетекция объектов...")
        results = model(frame)
        result = results[0]

        # Получаем данные обнаружений
        boxes = result.boxes
        names = result.names

        print(f"\nОбнаружено объектов: {len(boxes)}")
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = names[cls_id]
            print(f"  {name:20s} — {conf:.0%}")

            # Рисуем зеленый прямоугольник поверх оригинального (цветного) кадра
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Добавим подпись
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Вывод текущего кадра в окно
        cv2.imshow('Webcam Live', frame)

        # waitKey(1) ждет 1 миллисекунду нажатия клавиши.
        # Если нажата 'q', выходим из цикла
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Обязательно освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
