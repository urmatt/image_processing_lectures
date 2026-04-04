import cv2

def main():
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

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Находим лица
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Вырезаем область лица (Region of Interest)
            # В numpy массивах сначала идет строка (Y), затем столбец (X)
            face_roi = frame[y:y+h, x:x+w]
            
            # Применяем сильное Гауссово размытие к лицу
            # Чем больше ядро размытия (99, 99), тем сильнее блюр
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            
            # Вставляем размытое лицо обратно в оригинальный кадр
            frame[y:y+h, x:x+w] = blurred_face
            
            # Опционально: можно нарисовать рамку, чтобы было видно, где был блюр
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Anonymous Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
