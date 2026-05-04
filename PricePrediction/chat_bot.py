import os
import joblib
import pandas as pd
from datetime import datetime
import google.generativeai as genai

# ================= CẤU HÌNH API =================
GENAI_API_KEY = "AIzaSyBsyWhH6fKsX9MFLp0JsTkUb309DNw9Dtk"
genai.configure(api_key=GENAI_API_KEY)

# TỰ ĐỘNG TÌM MODEL PHÙ HỢP (Chống lỗi 404 Not Found)
valid_models = []
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        valid_models.append(m.name.replace('models/', ''))

if not valid_models:
    print("Lỗi: API Key của bác không hợp lệ hoặc chưa được cấp quyền!")
    exit()

# Ưu tiên chọn bản Flash nhanh nhất, nếu không có thì lấy bản đầu tiên tìm được
best_model = 'gemini-1.5-flash' if 'gemini-1.5-flash' in valid_models else valid_models[0]
chat_model = genai.GenerativeModel(best_model)
# ================================================

def load_local_ai():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        model = joblib.load(os.path.join(current_dir, 'price_prediction_model.pkl'))
        prod_encoder = joblib.load(os.path.join(current_dir, 'product_encoder.pkl'))
        df = pd.read_csv(os.path.join(current_dir, 'market_prices.csv'))
        return model, prod_encoder, df
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file Model hoặc CSV.")
        return None, None, None

def smart_assistant():
    local_model, prod_encoder, df = load_local_ai()
    if local_model is None: return

    print("\n" + "="*50)
    print("🤖 TRỢ LÝ AI BÁO GIÁ (BẢN TỰ TÌM KIẾM)")
    print("="*50)
    
    user_input = input("\n👤 Người dùng: ").strip()
    print("\n[Hệ thống đang xử lý...]")
    
    # TÌM TỪ KHÓA
    extract_prompt = f"""
    Trích xuất tên món ăn, thịt, cá, rau củ người dùng muốn mua trong câu: "{user_input}"
    Chỉ trả về 1 từ khóa ngắn gọn, viết thường. Không giải thích.
    """
    keyword = chat_model.generate_content(extract_prompt).text.strip().lower()
    
    # TRA DATA LOCAL (File CSV)
    product_dict = {str(item).lower(): str(item) for item in prod_encoder.classes_}
    matched_products = [real_name for lower_name, real_name in product_dict.items() if keyword in lower_name]

    # =========================================================
    # TRƯỜNG HỢP 1: LOCAL KHÔNG CÓ -> ÉP GEMINI TỰ LÙNG GIÁ
    # =========================================================
    if not matched_products:
        print(f" -> [Local CSV không có. AI đang tự động lùng sục giá thị trường cho '{user_input}'...]")
        
        search_prompt = f"""
        Mặt hàng: "{user_input}"
        Hệ thống dữ liệu nội bộ không có giá của mặt hàng này. 
        Dựa vào kho dữ liệu khổng lồ của bạn về vật giá siêu thị/chợ tại Việt Nam, hãy tự động ước tính mức giá bán lẻ hiện hành cho món này.
        
        TRẢ LỜI CỰC KỲ NGẮN GỌN THEO ĐÚNG FORMAT:
        - {user_input.capitalize()}: [Khoảng giá ước tính] VNĐ/kg (Nguồn: AI tự ước tính)
        Tuyệt đối không giải thích dông dài hay gợi ý món khác.
        """
        reply = chat_model.generate_content(search_prompt).text
        
        print("\n" + "="*50)
        print("🤖 AI:")
        print(reply.strip())
        return

    # =========================================================
    # TRƯỜNG HỢP 2: LOCAL CÓ DATA -> RANDOM FOREST XỬ LÝ
    # =========================================================
    today = datetime.now()
    day = today.day
    day_of_week = today.weekday()
    price_report = {}

    for product_name in matched_products:
        product_history = df[df['product_name'] == product_name]
        if product_history.empty: continue
        
        latest_data = product_history.iloc[-1]
        prod_encoded = prod_encoder.transform([product_name])[0]
        
        question = pd.DataFrame(
            [[day, day_of_week, 0, prod_encoded, latest_data['price_hcm'], latest_data['price_hn']]], 
            columns=['day', 'day_of_week', 'category_encoded', 'product_encoded', 'price_hcm', 'price_hn']
        )
        
        predicted_price = local_model.predict(question)[0]
        price_report[product_name] = int(predicted_price)

    price_data_str = "\n".join([f"- {k}: {v:,} VNĐ/kg" for k, v in price_report.items()])

    final_prompt = f"""
    Khách hỏi: "{user_input}"
    Bảng giá:
    {price_data_str}
    
    Nhiệm vụ: Trả lời CỰC KỲ NGẮN GỌN. CHỈ BÁO GIÁ. Không dông dài, không chào mời.
    """
    
    final_reply = chat_model.generate_content(final_prompt).text
    
    print("\n" + "="*50)
    print("🤖 AI:")
    print(final_reply.strip())

if __name__ == "__main__":
    smart_assistant()