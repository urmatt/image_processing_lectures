import cv2

def main():
    # Загружаем предобученную модель для поиска лиц из встроенных файлов OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return

    print("Камера успешно запущена. Нажмите 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Зеркальное отображение

        # Для каскадов Хаара нужно перевести изображение в ЧБ формат
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Детекция лиц
        # scaleFactor=1.1 - во сколько раз уменьшается размер изображения на каждом масштабе
        # minNeighbors=5 - сколько минимум соседей должно быть у кандидата, чтобы его признали лицом
        # minSize=(30, 30) - минимальный размер лица
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Перебираем все найденные лица (x, y - верхний левый угол; w, h - ширина и высота)
        for (x, y, w, h) in faces:
            # Рисуем зеленый прямоугольник поверх оригинального (цветного) кадра
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Добавим подпись
            cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
