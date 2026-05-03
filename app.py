from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI(title="AgriTech AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("ImageDetection/keras_model.h5", compile=False)

with open("ImageDetection/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesFile(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/api/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = preprocess_image(contents)
        
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_idx = np.argmax(score)
        
        return {
            "success": True,
            "label": labels[class_idx],
            "confidence": float(np.max(score))
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/")
def home():
    return {"message": "AgriTech AI Server is running!"}