from datetime import datetime
import io
import os
from dotenv import load_dotenv
import json
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import google.generativeai as genai
import joblib
import numpy as np
from typing import List
import sqlite3
import pandas as pd
from PIL import Image
from pydantic import BaseModel
from rag_pipeline import (
    DocumentProcessor,
    RAGChatbot,
    VectorDatabase,
    extract_text_from_file,
)
import requests
from transformers import pipeline

load_dotenv()

app = FastAPI(title="AgriTech All-in-One AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
  image_classifier = pipeline(
      "zero-shot-image-classification", model="openai/clip-vit-base-patch32"
  )
  DEFAULT_LABELS = [
      "vegetables",
      "fruits",
      "raw meat",
      "seafood and fish",
      "rice grains, roasted coffee beans, nuts, and seeds",
      "non-food items, objects, vehicles, or people",
  ]
  print("Đã nạp mô hình CLIP thành công!")
except Exception as e:
  print(f"Lỗi tải mô hình CLIP: {e}")

API_KEY = os.getenv("GENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
if API_KEY:
  genai.configure(api_key=API_KEY)
  valid_models = [
      m.name.replace("models/", "")
      for m in genai.list_models()
      if "generateContent" in m.supported_generation_methods
  ]
  best_model = (
      "gemini-1.5-flash"
      if "gemini-1.5-flash" in valid_models
      else valid_models[0]
  )
  gemini_price_model = genai.GenerativeModel(best_model)
else:
  gemini_price_model = None
  print("Chưa cấu hình API Key cho Gemini fallback dự đoán giá!")

PRICE_DIR = os.path.join(BASE_DIR, "PricePrediction")
try:
  price_model = joblib.load(
      os.path.join(PRICE_DIR, "price_prediction_model.pkl")
  )
  prod_encoder = joblib.load(os.path.join(PRICE_DIR, "product_encoder.pkl"))
  df_prices = pd.read_csv(os.path.join(PRICE_DIR, "market_prices.csv"))
  product_dict = {
      str(item).lower(): str(item) for item in prod_encoder.classes_
  }
  print("Đã nạp mô hình Dự đoán giá thành công!")
except Exception as e:
  print(f"Lỗi tải mô hình dự đoán giá: {e}")

vector_db = VectorDatabase()
chatbot = RAGChatbot(model_name="qwen2.5:3b") 
processor = DocumentProcessor(max_words=250, overlap_words=30)

class PriceQueryRequest(BaseModel):
  product_name: str

class ProductDescriptionRequest(BaseModel):
  product_name: str
  category: str

class ProductAdviceRequest(BaseModel):
  query: str
  product_name: str
  user_id: str = "guest"
  session_id: str | None = None

class ReviewSummarizeRequest(BaseModel):
    product_name: str
    reviews: List[str]

def query_qwen_for_summary(prompt: str, model_name: str = "qwen2.5:3b") -> str:
    """Gọi Ollama nội bộ để sinh JSON tóm tắt đánh giá"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            },
            timeout=25
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        return ""
    except Exception as e:
        print(f"Lỗi gọi Ollama: {e}")
        return ""

@app.get("/")
def home():
  return {"message": "AgriTech Unified AI Server is running!"}


@app.post("/api/predict-image")
async def predict_image(file: UploadFile = File(...)):
  try:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    results = image_classifier(image, candidate_labels=DEFAULT_LABELS)

    best_match_en = results[0]["label"]
    confidence = float(results[0]["score"])

    if best_match_en == "non-food items, objects, vehicles, or people":
      return {
          "success": False,
          "error": "Ảnh không hợp lệ. Vui lòng chụp đúng nông sản/thực phẩm.",
      }

    return {"success": True, "label": best_match_en, "confidence": confidence}
  except Exception as e:
    return {"success": False, "error": str(e)}


@app.post("/api/predict-price")
async def predict_price(request: PriceQueryRequest):
  try:
    keyword = request.product_name.strip().lower()
    matched_products = [
        real_name
        for lower_name, real_name in product_dict.items()
        if keyword in lower_name
    ]

    if not matched_products:
      if not gemini_price_model:
        return {
            "success": False,
            "error": "Không tìm thấy nông sản và chưa cấu hình Gemini fallback.",
        }

      search_prompt = f"""
            Đánh giá từ khóa: '{keyword}'
            Quy tắc 1: Nếu từ khóa KHÔNG PHẢI là thực phẩm, nông sản, thịt cá, gia vị hoặc rau củ, BẮT BUỘC trả về đúng 1 chữ: INVALID.
            Quy tắc 2: Nếu là thực phẩm/nông sản, hãy ước tính giá bán lẻ tại Việt Nam và CHỈ trả về con số (ví dụ: 50000 hoặc 40000-50000). Không giải thích thêm.
            """
      reply = gemini_price_model.generate_content(search_prompt).text.strip()

      if "INVALID" in reply.upper():
        return {
            "success": False,
            "error": (
                "Từ khóa không hợp lệ. Vui lòng chỉ nhập tên nông sản, thực"
                " phẩm."
            ),
        }

      return {"success": True, "product_name": keyword, "price": reply}

    today = datetime.now()
    best_match = matched_products[0]
    product_history = df_prices[df_prices["product_name"] == best_match]
    latest_data = product_history.iloc[-1]
    prod_encoded = prod_encoder.transform([best_match])[0]

    question = pd.DataFrame(
        [[
            today.day,
            today.weekday(),
            0,
            prod_encoded,
            latest_data["price_hcm"],
            latest_data["price_hn"],
        ]],
        columns=[
            "day",
            "day_of_week",
            "category_encoded",
            "product_encoded",
            "price_hcm",
            "price_hn",
        ],
    )
    predicted_price = int(price_model.predict(question)[0])

    return {
        "success": True,
        "product_name": best_match,
        "price": str(predicted_price),
    }
  except Exception as e:
    return {"success": False, "error": str(e)}


@app.post("/api/chat")
async def chat_endpoint(request: ProductAdviceRequest):
  try:
    session_id = (
        request.session_id or f"{request.user_id}_{request.product_name}"
    )
    search_query = f"{request.product_name}: {request.query}"

    knowledge_context = vector_db.search_context(
        query=search_query, product_name=request.product_name, top_k=5
    )

    answer = chatbot.generate_answer(
        query=request.query,
        product_name=request.product_name,
        context=knowledge_context,
        session_id=session_id,
    )
    return {
        "status": "success",
        "session_id": session_id,
        "product_name": request.product_name,
        "answer": answer,
        "context_used": knowledge_context,
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ProductAdviceRequest):
  session_id = request.session_id or f"{request.user_id}_{request.product_name}"
  search_query = f"{request.product_name}: {request.query}"

  knowledge_context = vector_db.search_context(
      query=search_query, product_name=request.product_name, top_k=5
  )

  def token_generator():
    for text_token in chatbot.generate_answer_stream(
        query=request.query,
        product_name=request.product_name,
        context=knowledge_context,
        session_id=session_id,
    ):
      yield text_token

  return StreamingResponse(
      token_generator(), media_type="text/plain; charset=utf-8"
  )


@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str):
  messages = chatbot.db.get_messages(session_id)
  return {"status": "success", "session_id": session_id, "messages": messages}


@app.delete("/api/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
  chatbot.db.clear_session(session_id)
  return {"status": "success", "message": "Đã xóa lịch sử trò chuyện"}


@app.post("/api/upload-knowledge")
async def upload_knowledge(file: UploadFile = File(...)):
  try:
    allowed_extensions = (".txt", ".md", ".pdf", ".docx")
    if not file.filename.lower().endswith(allowed_extensions):
      raise HTTPException(
          status_code=400,
          detail="Chỉ hỗ trợ các định dạng tài liệu: .txt, .md, .pdf, .docx",
      )

    raw_bytes = await file.read()
    text_content = extract_text_from_file(raw_bytes, file.filename)

    if not text_content.strip():
      raise HTTPException(
          status_code=400, detail="Không tìm thấy nội dung văn bản trong file."
      )

    chunks_data = processor.chunk_markdown(text_content)
    clean_name = file.filename.rsplit(".", 1)[0]
    vector_db.add_markdown_chunks(chunks_data, source_name=clean_name)

    return {
        "status": "success",
        "message": f"Đã nạp '{file.filename}' vào cơ sở dữ liệu!",
        "chunks_added": len(chunks_data),
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge-files")
def get_indexed_files():
  try:
    collection = vector_db.collection
    data = collection.get(include=["metadatas"])

    counter = Counter(
        meta.get("source") or meta.get("source_name", "Không rõ")
        for meta in data.get("metadatas", [])
        if meta
    )

    return {
        "status": "success",
        "total_chunks": collection.count(),
        "files": [
            {"source": filename, "chunks": count}
            for filename, count in counter.items()
        ],
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-description")
async def generate_product_description(request: ProductDescriptionRequest):
  try:
    user_prompt = (
        f"Tên sản phẩm: {request.product_name}\nDanh mục: {request.category}"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "agritech-desc",
            "prompt": user_prompt,
            "stream": False,
        },
        timeout=60,
    )

    if response.status_code != 200:
      raise HTTPException(
          status_code=500, detail=f"Lỗi từ Ollama Service: {response.text}"
      )

    data = response.json()
    description = data.get("response", "").strip()

    return {
        "success": True,
        "product_name": request.product_name,
        "category": request.category,
        "description": description,
    }
  except requests.exceptions.ConnectionError:
    raise HTTPException(
        status_code=503,
        detail=(
            "Không thể kết nối đến Ollama. Hãy chắc chắn Ollama đang chạy trên"
            " máy (port 11434)."
        ),
    )
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-description/stream")
async def generate_product_description_stream(
    request: ProductDescriptionRequest,
):
  user_prompt = (
      f"Tên sản phẩm: {request.product_name}\nDanh mục: {request.category}"
  )

  def stream_generator():
    try:
      with requests.post(
          "http://localhost:11434/api/generate",
          json={
              "model": "agritech-desc",
              "prompt": user_prompt,
              "stream": True,
          },
          stream=True,
          timeout=60,
      ) as res:
        for line in res.iter_lines():
          if line:
            chunk = json.loads(line)
            yield chunk.get("response", "")
    except Exception as e:
      yield f"\n[Lỗi kết nối Ollama: {str(e)}]"

  return StreamingResponse(
      stream_generator(), media_type="text/plain; charset=utf-8"
  )

@app.post("/api/summarize-reviews")
async def summarize_reviews_endpoint(request: ReviewSummarizeRequest):
    if not request.reviews:
        return {"success": False, "error": "Chưa có đánh giá nào để tóm tắt."}

    review_text = "\n".join([f"- {r}" for r in request.reviews[:15]])

    prompt = f"""<|im_start|>system
Bạn là trợ lý sàn nông sản AgriTech. Hãy đọc các đánh giá của khách hàng về sản phẩm '{request.product_name}' và phân loại chính xác thành ƯU ĐIỂM và NHƯỢC ĐIỂM thực tế.

Quy tắc:
1. Chỉ dựa trên phản hồi có thật của khách, tuyệt đối không tự bịa đặt.
2. Mỗi ý phải ngắn gọn, súc tích (dưới 15 từ).
3. Số lượng ý: Trích xuất từ 1 đến tối đa 4 ý thực tế nhất cho mỗi mục. Nếu khách hàng không chê hoặc không có điểm lưu ý nào, mục "cons" BẮT BUỘC để mảng rỗng [].
4. BẮT BUỘC trả về đúng định dạng JSON sau:
{{
  "pros": [
    "Ý điểm mạnh thực tế trích từ đánh giá (tối đa 4 ý)"
  ],
  "cons": [
    "Ý điểm yếu/lưu ý thực tế (để trống [] nếu không ai chê)"
  ]
}}
Không viết thêm văn bản giải thích nào khác ngoài chuỗi JSON trên.
<|im_end|>
<|im_start|>user
Danh sách đánh giá:
{review_text}
<|im_end|>
<|im_start|>assistant
"""
    raw_output = query_qwen_for_summary(prompt, model_name="qwen2.5:3b")

    try:
        start_idx = raw_output.find('{')
        end_idx = raw_output.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = raw_output[start_idx:end_idx]
            data = json.loads(json_str)
            return {
                "success": True,
                "pros": data.get("pros", []),
                "cons": data.get("cons", [])
            }
        return {"success": False, "error": "AI không trả về JSON đúng cấu trúc."}
    except Exception as e:
        return {"success": False, "error": f"Lỗi parse JSON: {str(e)}"}

if __name__ == "__main__":
  import uvicorn

  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)