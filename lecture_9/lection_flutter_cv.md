# Лекция 9. Интеграция моделей компьютерного зрения в мобильные приложения (Flutter)

В предыдущих лекциях мы научились создавать и использовать мощные нейросетевые модели на Python, такие как YOLOv8. Однако, конечным пользователям редко нужен скрипт на Python — им нужно удобное мобильное или веб-приложение.

В этой лекции мы разберем, как подключить нашу модель YOLOv8 к мобильному приложению на Flutter.

## Архитектура: Клиент-Сервер

Существует два основных способа запустить модель в мобильном приложении:
1. **On-Device Inference (Запуск на устройстве):** Модель конвертируется в специальные мобильные форматы (например, TFLite или CoreML) и запускается прямо на процессоре/видеокарте телефона. Это сложнее, требует оптимизации модели под мобильные устройства и увеличивает размер приложения, но работает без интернета.
2. **Клиент-Сервер (Backend-Frontend):** Модель работает на мощном сервере, а мобильное приложение просто отправляет фотографии на этот сервер по HTTP-запросу и получает готовый результат. Это стандартный подход для тяжелых моделей (таких как YOLOv8 или ChatGPT).

В этой лекции мы реализуем **второй подход**.

### 1. Backend (Python + FastAPI)

Для начала нам нужен сервер, который будет принимать картинки. Мы будем использовать микрофреймворк `FastAPI`, так как он работает очень быстро и просто настраивается.

Установим нужные библиотеки:
```bash
pip install fastapi uvicorn python-multipart ultralytics opencv-python
```

#### Код сервера (`backend/main.py`):
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO

# Создаем приложение FastAPI
app = FastAPI(title="YOLOv8 API")

# Загружаем модель сегментации (из прошлой лекции)
model = YOLO('../../yolov8n-seg.pt')

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # 1. Читаем картинку из отправленных байтов
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Передаем изображение в модель YOLOv8
    results = model(img)
    
    # 3. Модель сама рисует рамки и маски на картинке
    annotated_img = results[0].plot()

    # 4. Конвертируем картинку обратно в формат JPEG для отправки на телефон
    _, encoded_img = cv2.imencode('.jpg', annotated_img)
    
    # 5. Возвращаем картинку
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    # Запуск сервера на порту 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Запустить сервер можно командой: `python main.py`

### 2. Frontend (Flutter)

Теперь создадим мобильное приложение, которое будет делать фотографии и отправлять их на сервер.

Создаем проект и добавляем пакеты:
```bash
flutter create flutter_cv_app
cd flutter_cv_app
flutter pub add http image_picker
```
- `image_picker` нужен для того, чтобы открывать галерею телефона или камеру.
- `http` нужен для отправки запросов на наш FastAPI сервер.

#### Основной код отправки изображения:
Чтобы отправить файл (картинку) на сервер, мы используем специальный тип HTTP-запроса, который называется **Multipart Request**.

```dart
import 'package:http/http.dart' as http;

Future<void> _processImage() async {
  // IP адрес вашего компьютера (сервера)
  // Для эмулятора Android это обычно 10.0.2.2, для компьютера (macOS/Win) локально — 127.0.0.1
  final url = Uri.parse('http://127.0.0.1:8000/predict');
  
  // Создаем Multipart POST запрос
  final request = http.MultipartRequest('POST', url)
    ..files.add(await http.MultipartFile.fromPath('file', _image!.path));

  // Отправляем на сервер
  final response = await request.send();

  if (response.statusCode == 200) {
    // Получаем байты готовой картинки с нарисованными масками
    final bytes = await response.stream.toBytes();
    setState(() {
      _processedImage = bytes;
    });
  }
}
```

Для отображения загруженной из интернета (или в виде байтов) картинки во Flutter используется виджет `Image.memory`:
```dart
Image.memory(_processedImage!, height: 300, fit: BoxFit.contain)
```

> [!NOTE]
> В папке `flutter_cv_app` вы найдете полный код рабочего приложения, в котором реализован выбор картинки, загрузка и обработка состояния.

### Задание для самостоятельной работы:
1. Запустите Python-сервер.
2. Запустите Flutter-приложение (на телефоне или эмуляторе/десктопе).
3. Измените модель на сервере с сегментации (`yolov8n-seg.pt`) на детекцию обычных рамок (`yolov8n.pt`). Обратите внимание, что код Flutter-приложения при этом менять не нужно — это главное преимущество клиент-серверной архитектуры!
