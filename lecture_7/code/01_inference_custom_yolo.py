import cv2
from ultralytics import YOLO
import math

def main():
    # 1. Загрузка НАШЕЙ обученной модели. 
    # Замените 'best.pt' на путь к вашему файлу весов, который вы скачали из Colab!
    model_path = 'best.pt' 
    
    try:
        model = YOLO(model_path)
        print(f"Успешно загружена кастомная модель: {model_path}")
    except Exception as e:
        print(f"Ошибка загрузки модели. Проверьте, есть ли файл {model_path} в папке.")
        print(e)
        return

    # 2. Инициализация веб-камеры
    cap = cv2.VideoCapture(0)

    # Установим разрешение повыше (опционально)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть веб-камеру")
        return

    print("Захват видео начался. Нажмите 'q' для выхода.")

    # 3. Бесконечный цикл обработки(По кадрам)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Зеркальное отображение (чтобы было удобнее смотреть на себя)
        frame = cv2.flip(frame, 1)

        # 4. Пропускаем кадр через нашу обученную нейросеть
        # conf=0.5 означает, что мы берем только те объекты, в которых сеть уверена на 50% и выше
        results = model(frame, conf=0.5, verbose=False)

        # Извлекаем данные о найденных объектах
        boxes = results[0].boxes

        for box in boxes:
            # Получаем координаты ограничивающей рамки (Bounding Box)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Получаем уверенность (Confidence) сети в формате от 0.0 до 1.0 -> в проценты
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Получаем номер класса (ID). Для кастомной модели, если был 1 класс, это 0
            cls = int(box.cls[0])
            
            # Получаем название класса по его ID
            class_name = model.names[cls]

            # 5. Отрисовка результатов на кадре
            # Нарисуем прямоугольник вокруг объекта (Фиолетового цвета)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Формируем текст: Название класса и уверенность
            label = f'{class_name} {conf:.2f}'

            # Рассчитываем размер текста для красивой подложки
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Рисуем подложку (черного цвета), чтобы текст лучше читался на любом фоне
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), (255, 0, 255), cv2.FILLED)
            
            # Пишем сам белый текст поверх подложки
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Практическое дополнение: если сеть нашла ваш объект, можно вывести сообщение в терминал!
            # Например: если вы обучали модель находить ваш пропуск
            # if class_name == "Id_card":
            #     print("Доступ разрешен!")

        # Вывод кадра на экран
        cv2.imshow("Custom Object Detection System", frame)

        # Выход по клавише 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Очистка ресурсов
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
