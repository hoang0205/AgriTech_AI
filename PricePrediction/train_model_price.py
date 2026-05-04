import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os
import joblib

def train_price_predictor():
    print("1. Đang đọc dữ liệu từ market_prices.csv...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_path = os.path.join(current_dir, 'market_prices.csv')
    
    df = pd.read_csv(csv_path)

    print("2. Đang dọn dẹp dữ liệu...")
    df.replace(0, np.nan, inplace=True)
    
    df.dropna(inplace=True)

    print("3. Đang mã hóa dữ liệu...")
    label_encoder_cat = LabelEncoder()
    df['category_encoded'] = label_encoder_cat.fit_transform(df['category'])
    
    label_encoder_prod = LabelEncoder()
    df['product_encoded'] = label_encoder_prod.fit_transform(df['product_name'])

    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek

    print("4. Đang chia tách dữ liệu...")
    X = df[['day', 'day_of_week', 'category_encoded', 'product_encoded', 'price_hcm', 'price_hn']]
    
    y = df['market_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("5. Bắt đầu huấn luyện AI...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("6. Test AI.")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    score = r2_score(y_test, predictions)

    print("\n================ KẾT QUẢ ================")
    print(f"Độ chính xác của mô hình: {score * 100:.2f}%")
    print(f"Sai số trung bình khi dự đoán: {mae:,.0f} VNĐ")
    
    print("\n--- Thử nghiệm thực tế  ---")
    results = pd.DataFrame({
        'Giá thực tế': y_test.values[:5],
        'AI Dự đoán': predictions[:5].round()
    })
    print(results)

    print("\n7. Đang lưu mô hình...")
    joblib.dump(model, os.path.join(current_dir, 'price_prediction_model.pkl'))

    joblib.dump(label_encoder_cat, os.path.join(current_dir, 'category_encoder.pkl'))
    joblib.dump(label_encoder_prod, os.path.join(current_dir, 'product_encoder.pkl'))
    
    print("Đã lưu")

if __name__ == "__main__":
    train_price_predictor()