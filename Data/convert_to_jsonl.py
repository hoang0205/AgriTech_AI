import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split

def convert_csv_to_jsonl():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "nong_san_augmented.csv")
    
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file {input_file}. Hãy chạy file processing.py trước!")
        return

    df = pd.read_csv(input_file)
    print(f"Đã nạp {len(df)} mẫu dữ liệu siêu sạch vào bộ chia.")

    system_instruction = (
        "Bạn là một chuyên gia viết nội dung bán hàng nông sản thông minh. "
        "Hãy viết một đoạn mô tả sản phẩm thật hấp dẫn, chi tiết và chuẩn SEO dựa trên tên và danh mục sản phẩm."
    )

    dataset_records = []
    for _, row in df.iterrows():
        input_text = f"Tên sản phẩm: {row['Ten_SP']}\nDanh mục: {row['Danh_Muc']}"
        record = {
            "instruction": system_instruction,
            "input": input_text,
            "output": row['Mo_Ta']
        }
        dataset_records.append(record)

    train_data, val_data = train_test_split(dataset_records, test_size=0.1, random_state=42)

    def write_jsonl(data, file_name):
        file_path = os.path.join(current_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Đã xuất: {file_name} ({len(data)} mẫu)")

    write_jsonl(train_data, "nong_san_train.jsonl")
    write_jsonl(val_data, "nong_san_val.jsonl")
    print("\n[THÀNH CÔNG] File JSONL đã sẵn sàng để đẩy lên Colab huấn luyện ngay!")

if __name__ == "__main__":
    convert_csv_to_jsonl()