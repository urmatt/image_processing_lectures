import cv2
from ultralytics import YOLO

def main():
    print("Загрузка модели YOLO11...")
    # Автоматически скачает веса, если не найдет локально
    model = YOLO("yolo11n.pt")  
    print("Модель загружена!")

    cap = cv2.VideoCapture(0)

    # Инициализация детектора штрих-кодов (встроено в современные версии OpenCV)
    if hasattr(cv2, 'barcode'):
        barcode_detector = cv2.barcode.BarcodeDetector()
    else:
        barcode_detector = None

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return

    print("Камера успешно запущена. Нажмите 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Детекция (отключаем лишний вывод в консоль на каждом кадре)
        results = model(frame, verbose=False)
        result = results[0]
        boxes = result.boxes
        names = result.names

        bottles = []
        caps = []

        # ----------------------------------------------------
        # 1. Распределение найденных объектов (Бутылки и Крышки)
        # ----------------------------------------------------
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = names[cls_id].lower()

            # Снижаем порог уверенности (Confidence), чтобы ловить лежащие (горизонтальные) 
            # бутылки, в которых нейросеть обычно сомневается из-за "смещения данных" (Data bias)
            if conf > 0.25:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if name in ['bottle', 'bottle_with_cap', 'bottle_without_cap']:
                    bottles.append((x1, y1, x2, y2, name))
                elif name in ['bottle cap', 'bottle_cap', 'cap']:
                    caps.append((x1, y1, x2, y2))

        # ----------------------------------------------------
        # 2. Отрисовка найденных отдельных крышек
        # ----------------------------------------------------
        for (cx1, cy1, cx2, cy2) in caps:
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
            cv2.putText(frame, "Cap", (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # ----------------------------------------------------
        # 3. Анализ бутылок и их статуса (С крышкой / Без)
        # ----------------------------------------------------
        for (bx1, by1, bx2, by2, bottle_name) in bottles:
            bw = bx2 - bx1
            bh = by2 - by1
            if bw == 0: continue
            
            # Анализ "Форм-фактора" бутылки (отношение высоты к ширине)
            aspect_ratio = bh / bw
            if aspect_ratio >= 1.5:
                label = f"{bottle_name} (Ratio: {aspect_ratio:.1f})"
                box_color = (0, 255, 0)
            else:
                label = f"{bottle_name}? (Ratio: {aspect_ratio:.1f})"
                box_color = (0, 165, 255)

            # Отрисовка основной рамки бутылки
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
            cv2.putText(frame, label, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Если модель САМА выдала класс с/без крышки, сразу показываем этот статус:
            if bottle_name == 'bottle_with_cap':
                cv2.putText(frame, "STATUS: CLOSED (Has Cap)", (bx1, by1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                continue
            elif bottle_name == 'bottle_without_cap':
                cv2.putText(frame, "STATUS: OPEN (No Cap)", (bx1, by1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue

            # Для обычного класса 'bottle' (когда статус неизвестен):
            # Проверяем, "надета" ли найденная отдельная крышка на эту бутылку
            has_cap = False
            for (cx1, cy1, cx2, cy2) in caps:
                cap_center_x = (cx1 + cx2) / 2
                cap_center_y = (cy1 + cy2) / 2
                
                # Крышка считается надетой, если её центр находится внутри или очень 
                # близко к границам бутылки (не ограничиваемся только верхней гранью)
                margin_x = bw * 0.15
                margin_y = bh * 0.15
                if (bx1 - margin_x) <= cap_center_x <= (bx2 + margin_x) and \
                   (by1 - margin_y) <= cap_center_y <= (by2 + margin_y):
                    has_cap = True
                    break
            
            # ЭВРИСТИКА ФОЛЛБЕК ПРОТИВ СТАНДАРТНЫХ МОДЕЛЕЙ:
            if not has_cap:
                # Если бутылка стоит вертикально (высота > ширина)
                if bh >= bw:
                    cap_h_est, cap_w_est = int(bh * 0.15), int(bw * 0.45)
                    cx1 = max(0, bx1 + int((bw - cap_w_est) / 2))
                    cx2 = min(frame.shape[1], cx1 + cap_w_est)
                    cy1 = max(0, by1)
                    cy2 = min(frame.shape[0], by1 + cap_h_est)
                else:
                    # Бутылка лежит горизонтально (крышка сбоку)
                    cap_h_est, cap_w_est = int(bh * 0.45), int(bw * 0.15)
                    cy1 = max(0, by1 + int((bh - cap_h_est) / 2))
                    cy2 = min(frame.shape[0], cy1 + cap_h_est)
                    # Проверяем левую грань (можно проверять и правую, для простоты берём начало горлышка)
                    cx1 = max(0, bx1)
                    cx2 = min(frame.shape[1], bx1 + cap_w_est)

                if cx2 > cx1 and cy2 > cy1:
                    cap_roi = frame[cy1:cy2, cx1:cx2]
                    hsv_roi = cv2.cvtColor(cap_roi, cv2.COLOR_BGR2HSV)
                    gray_roi = cv2.cvtColor(cap_roi, cv2.COLOR_BGR2GRAY)
                    
                    if hsv_roi[:, :, 1].mean() > 40 or (5 < gray_roi.std() < 40):
                        has_cap = True
                        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
                        cv2.putText(frame, "Cap (Heur)", (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if has_cap:
                cv2.putText(frame, "STATUS: CLOSED (Has Cap)", (bx1, by1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "STATUS: OPEN (No Cap)", (bx1, by1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ----------------------------------------------------
        # 3. Поиск штрих-кодов на кадре
        # ----------------------------------------------------
        if barcode_detector is not None:
            retval, decoded_info, points, _ = barcode_detector.detectAndDecodeMulti(frame)
            if retval and points is not None:
                for i, info in enumerate(decoded_info):
                    pts = points[i].astype(int)
                    # Отрисовка контура штрих-кода красным цветом
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    
                    # Пишем текст штрих-кода (или просто Barcode, если значение не расшифровано)
                    text = f"Barcode: {info}" if info else "Barcode"
                    cv2.putText(frame, text, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Bottle & Cap Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
