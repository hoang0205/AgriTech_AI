import joblib
import pandas as pd
import os
from datetime import datetime

def predict_future_price():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        model = joblib.load(os.path.join(current_dir, 'price_prediction_model.pkl'))
        prod_encoder = joblib.load(os.path.join(current_dir, 'product_encoder.pkl'))
        df = pd.read_csv(os.path.join(current_dir, 'market_prices.csv'))
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file Model hoặc CSV.")
        return

    print("\n--- 🤖 TRỢ LÝ AI BÁO GIÁ THÔNG MINH ---")
    
    user_product = input("Nhập tên món đồ muốn mua (VD: bò, gà, cá, rau): ").strip().lower()
    
    product_dict = {str(item).lower(): str(item) for item in prod_encoder.classes_}
    matched_products = [real_name for lower_name, real_name in product_dict.items() if user_product in lower_name]

    if len(matched_products) == 0:
        print(f"\n[X] Chịu, chợ hôm nay không có món nào tên là '{user_product}' cả.")
        return

    today = datetime.now()
    day = today.day
    day_of_week = today.weekday()
    
    # Tạo một dictionary để lưu giá của từng mặt hàng
    price_report = {}

    for product_name in matched_products:
        product_history = df[df['product_name'] == product_name]
        if product_history.empty: continue
            
        latest_data = product_history.iloc[-1]
        auto_price_hcm = latest_data['price_hcm']
        auto_price_hn = latest_data['price_hn']

        prod_encoded = prod_encoder.transform([product_name])[0]
        cat_encoded = 0 
        
        question = pd.DataFrame(
            [[day, day_of_week, cat_encoded, prod_encoded, auto_price_hcm, auto_price_hn]], 
            columns=['day', 'day_of_week', 'category_encoded', 'product_encoded', 'price_hcm', 'price_hn']
        )
        
        predicted_price = model.predict(question)[0]
        price_report[product_name] = predicted_price

    if not price_report:
        print("Dữ liệu bị lỗi, không thể dự đoán.")
        return

    # Sắp xếp từ rẻ nhất đến đắt nhất
    sorted_report = sorted(price_report.items(), key=lambda x: x[1])
    
    min_price = sorted_report[0][1]
    max_price = sorted_report[-1][1]

    print("\n================= KẾT QUẢ TƯ VẤN =================")
    print(f"Bác muốn mua '{user_product.upper()}' loại nào?")
    
    if len(sorted_report) > 1:
        print(f"👉 Khoảng giá dao động: {min_price:,.0f}đ  đến  {max_price:,.0f}đ /kg")
        print("\n📋 Bảng giá dự đoán chi tiết từng loại (Từ rẻ đến đắt):")
    else:
        print("\n📋 Bảng giá dự đoán chi tiết:")

    for name, price in sorted_report:
        # Nếu có từ khóa tìm kiếm, bôi đậm nó lên cho dễ nhìn (ở đây in hoa)
        print(f"  + {name}: {price:,.0f} VNĐ/kg")
        
    print("==================================================")

if __name__ == "__main__":
    predict_future_price()