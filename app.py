import os
import io
import requests
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# THÊM THƯ VIỆN CỦA HUGGING FACE CHO CLIP
from transformers import pipeline

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

print("Đang tải Zero-shot model (CLIP)... Quá trình này sẽ hơi lâu ở lần chạy đầu tiên!")
try:
    image_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
    
    DEFAULT_LABELS = [
    "vegetables",                                        
    "fruits",                                          
    "raw meat",                                         
    "seafood and fish",                                  
    "agricultural products like rice, beans, or coffee",  
    "non-food items, objects, vehicles, or people"      
]
    print("Đã nạp thành công mô hình Nhận diện ảnh CLIP!")
except Exception as e:
    print(f"Lỗi tải mô hình CLIP: {e}")

API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    print("Chưa tìm thấy file .env!")
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
    print("Đã nạp thành công mô hình Dự đoán giá!")
except Exception as e:
    print(f"Lỗi tải mô hình dự đoán giá: {e}")

class ImageRequest(BaseModel):
    image_url: str

class PriceQueryRequest(BaseModel):
    product_name: str

@app.get("/")
def home():
    return {"message": "AgriTech AI Server is running (Zero-Shot CLIP + Price Prediction)!"}

@app.post("/api/predict-image")
async def predict_image(request: ImageRequest):
    try:
        response = requests.get(request.image_url)
        response.raise_for_status() 
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        results = image_classifier(image, candidate_labels=DEFAULT_LABELS)
        
        best_match = results[0]
        
        return {
            "success": True,
            "label": best_match['label'],
            "confidence": float(best_match['score'])
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/predict-price")
async def predict_price(request: PriceQueryRequest):
    try:
        keyword = request.product_name.strip().lower()
        
        matched_products = [real_name for lower_name, real_name in product_dict.items() if keyword in lower_name]

        if not matched_products:
            search_prompt = f"""
            Đánh giá từ khóa: '{keyword}'
            Quy tắc 1 (Ưu tiên cao nhất): Nếu từ khóa KHÔNG PHẢI là thực phẩm, nông sản, thịt cá, gia vị hoặc rau củ (ví dụ: điện thoại, quần áo, xe máy, đồ vật, tên riêng...), BẮT BUỘC trả về đúng 1 chữ: INVALID.
            Quy tắc 2: NẾU VÀ CHỈ NẾU từ khóa là thực phẩm/nông sản, hãy ước tính giá bán lẻ tại Việt Nam và CHỈ trả về con số (ví dụ: 50000 hoặc 40000-50000).
            Tuyệt đối không giải thích, không in thêm chữ nào khác.
            """
            reply = chat_model.generate_content(search_prompt).text.strip()

            if "INVALID" in reply.upper():
                return {
                    "success": False,
                    "error": "Từ khóa không hợp lệ. Vui lòng chỉ nhập tên nông sản, thực phẩm."
                }

            return {
                "success": True,
                "product_name": keyword,
                "price": reply 
            }

        today = datetime.now()
        best_match = matched_products[0]
        
        product_history = df_prices[df_prices['product_name'] == best_match]
        latest_data = product_history.iloc[-1]
        prod_encoded = prod_encoder.transform([best_match])[0]
        
        question = pd.DataFrame(
            [[today.day, today.weekday(), 0, prod_encoded, latest_data['price_hcm'], latest_data['price_hn']]], 
            columns=['day', 'day_of_week', 'category_encoded', 'product_encoded', 'price_hcm', 'price_hn']
        )
        
        predicted_price = int(price_model.predict(question)[0])
        
        return {
            "success": True,
            "product_name": best_match,
            "price": str(predicted_price) 
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}