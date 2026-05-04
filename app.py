import os
import io
import requests
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AgriTech AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


try:
    image_model_path = os.path.join(BASE_DIR, "ImageDetection", "keras_model.h5")
    label_path = os.path.join(BASE_DIR, "ImageDetection", "labels.txt")
    
    image_model = tf.keras.models.load_model(image_model_path, compile=False)
    with open(label_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    print("✅ Đã nạp thành công mô hình Nhận diện ảnh!")
except Exception as e:
    print(f"❌ Lỗi tải mô hình Nhận diện ảnh: {e}")

# Setup Gemini
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    print("❌ CẢNH BÁO: Chưa tìm thấy GENAI_API_KEY trong file .env!")
else:
    genai.configure(api_key=API_KEY)
    valid_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    best_model = 'gemini-1.5-flash' if 'gemini-1.5-flash' in valid_models else valid_models[0]
    chat_model = genai.GenerativeModel(best_model)

PRICE_DIR = os.path.join(BASE_DIR, 'PricePrediction')
try:
    price_model = joblib.load(os.path.join(PRICE_DIR, 'price_prediction_model.pkl'))
    prod_encoder = joblib.load(os.path.join(PRICE_DIR, 'product_encoder.pkl'))
    df_prices = pd.read_csv(os.path.join(PRICE_DIR, 'market_prices.csv'))
    product_dict = {str(item).lower(): str(item) for item in prod_encoder.classes_}
    print("✅ Đã nạp thành công mô hình Dự đoán giá!")
except Exception as e:
    print(f"❌ Lỗi tải mô hình Dự đoán giá: {e}")



class ImageRequest(BaseModel):
    image_url: str

class PriceQueryRequest(BaseModel):
    user_text: str

@app.get("/")
def home():
    return {"message": "AgriTech AI Server is running (Image Detection + Price Prediction)!"}

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/api/predict-image")
async def predict_image(request: ImageRequest):
    try:
        response = requests.get(request.image_url)
        response.raise_for_status() 
        contents = response.content
        
        img_array = preprocess_image(contents)
        
        predictions = image_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_idx = np.argmax(score)
        
        return {
            "success": True,
            "label": labels[class_idx],
            "confidence": float(np.max(score))
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/predict-price")
async def predict_price(request: PriceQueryRequest):
    try:
        user_input = request.user_text
        
        extract_prompt = f"Trích xuất tên món ăn, thịt, cá, rau củ trong câu: '{user_input}'. Trả về 1 từ khóa viết thường, không giải thích."
        keyword = chat_model.generate_content(extract_prompt).text.strip().lower()
        
        matched_products = [real_name for lower_name, real_name in product_dict.items() if keyword in lower_name]

        if not matched_products:
            search_prompt = f"Mặt hàng '{keyword}'. Tự ước tính mức giá bán lẻ hiện hành tại Việt Nam. Format: Tên món: [Giá] VNĐ/kg."
            reply = chat_model.generate_content(search_prompt).text.strip()
            return {
                "success": True,
                "source": "llm_estimation",
                "keyword": keyword,
                "data": [{"product_name": keyword, "predicted_price_vnd": reply}]
            }

        today = datetime.now()
        results = []

        for product_name in matched_products:
            product_history = df_prices[df_prices['product_name'] == product_name]
            if product_history.empty: continue
            
            latest_data = product_history.iloc[-1]
            prod_encoded = prod_encoder.transform([product_name])[0]
            
            question = pd.DataFrame(
                [[today.day, today.weekday(), 0, prod_encoded, latest_data['price_hcm'], latest_data['price_hn']]], 
                columns=['day', 'day_of_week', 'category_encoded', 'product_encoded', 'price_hcm', 'price_hn']
            )
            
            predicted_price = int(price_model.predict(question)[0])
            results.append({
                "product_name": product_name,
                "predicted_price_vnd": predicted_price
            })

        return {
            "success": True,
            "source": "local_model",
            "keyword": keyword,
            "data": results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}