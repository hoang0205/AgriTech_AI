import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

app = FastAPI(title="AgriTech AI Core API")

model = load_model("keras_model.h5", compile=False)

with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)
        
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = float(prediction[0][index])

        clean_class_name = class_name.split(" ", 1)[1] if " " in class_name else class_name

        THRESHOLD = 0.70 
        
        if confidence_score < THRESHOLD:
            conclusion = "Anh_Rac"
            notification = "Ảnh quá mờ hoặc vật thể lạ. Vui lòng chụp rõ nông sản!"
        elif clean_class_name == "Anh_Rac":
            conclusion = "Anh_Rac"
            notification = "Ảnh không hợp lệ. Hệ thống chỉ nhận Nông sản, Thịt, Hải sản!"
        else:
            conclusion = clean_class_name
            notification = f"Nhận diện thành công: {clean_class_name}"

        return {
            "category": conclusion,
            "reliability": round(confidence_score * 100, 2),
            "notification": notification
        }

    except Exception as e:
        return {"error": str(e)}