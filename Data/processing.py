import pandas as pd
import os
import re

def clean_duplicate_paragraphs(text):
    if not isinstance(text, str):
        return text
    paragraphs = text.split('\n')
    seen = set()
    result = []
    for p in paragraphs:
        p_clean = p.strip()
        if p_clean and p_clean not in seen:
            seen.add(p_clean)
            result.append(p_clean)
    return '\n'.join(result)

def process_and_clean_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "tong_hop_nong_san_clean.csv")
    
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f"Tổng số mẫu ban đầu (thô): {len(df)}")

    # 1. Bỏ mẫu "Đang cập nhật..."
    df_clean = df[df['Mo_Ta'].notna() & (df['Mo_Ta'] != "Đang cập nhật...")].copy()

    # 2. Xóa thương hiệu triệt để bằng Regex
    df_clean['Mo_Ta'] = df_clean['Mo_Ta'].apply(
        lambda x: re.sub(r'Bách h(óa|oá)\s*XANH', 'AgriTech', str(x), flags=re.IGNORECASE)
    )
    df_clean['Mo_Ta'] = df_clean['Mo_Ta'].apply(
        lambda x: re.sub(r'tại\s*AgriTech', '', str(x), flags=re.IGNORECASE)
    )

    # 3. Lọc câu lặp nội bộ (như lỗi của Bắp mỹ)
    df_clean['Mo_Ta'] = df_clean['Mo_Ta'].apply(clean_duplicate_paragraphs)

    # 4. BỘ LỌC TỪ COLAB: Lọc câu cụt lủn và kiểm tra độ dài
    invalid_endings = (',', '...', 'và', 'của', 'còn', '-', ':')
    # Lọc bỏ các dòng có đuôi kết thúc không hợp lệ
    df_clean = df_clean[~df_clean['Mo_Ta'].str.strip().str.endswith(invalid_endings)]
    
    # Lọc bỏ dòng quá ngắn (dưới 150 ký tự - chuẩn hơn mức 50 ký tự trên Colab)
    df_clean['Length'] = df_clean['Mo_Ta'].str.len()
    df_clean = df_clean[df_clean['Length'] >= 150]
    
    # Cắt gọn các bài siêu dài (vượt 1500 ký tự)
    df_clean['Mo_Ta'] = df_clean['Mo_Ta'].apply(
        lambda x: x[:1497] + '...' if len(x) > 1500 else x
    )
    df_clean = df_clean.drop(columns=['Length'])

    # 5. Lưu ra file CSV chuẩn mực
    cleaned_file_path = os.path.join(current_dir, "nong_san_ready_for_ai.csv")
    df_clean.to_csv(cleaned_file_path, index=False, encoding='utf-8-sig')

    print(f"Hoàn tất lọc! Số mẫu sạch tinh khiết: {len(df_clean)} / {len(df)}")
    print(f"Đã lưu tại: {cleaned_file_path}")

if __name__ == "__main__":
    process_and_clean_data()