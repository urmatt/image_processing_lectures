from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI(title="YOLOv8 Flutter API")

# Загружаем модель сегментации
model = YOLO('../../yolov8n-seg.pt')

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Читаем картинку из байтов
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Инференс модели YOLOv8
    results = model(img)
    
    # Отрисовка масок и рамок
    annotated_img = results[0].plot()

    # Конвертируем обратно в JPEG для отправки клиенту
    _, encoded_img = cv2.imencode('.jpg', annotated_img)
    
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
