import pandas as pd
import os
import time
import google.generativeai as genai

genai.configure(api_key="AIzaSyCG8YKTzqwezpg86dN5SogAun_p6iDiQ6Q")

model = genai.GenerativeModel('gemini-2.5-flash')

def augment_description(ten_sp, danh_muc, mo_ta_goc):
    """
    Hàm gọi API để viết lại mô tả sản phẩm thành 2 phiên bản khác nhau.
    Prompt được thiết kế cực kỳ chặt chẽ (Guardrails) để chống bịa đặt (hallucination).
    """
    prompt = f"""
    Bạn là một chuyên gia viết nội dung bán hàng xuất sắc trên sàn thương mại điện tử nông sản AgriTech.
    Dưới đây là thông tin gốc của một sản phẩm:
    - Tên sản phẩm: {ten_sp}
    - Danh mục: {danh_muc}
    - Mô tả gốc: {mo_ta_goc}

    NHIỆM VỤ CỦA BẠN:
    Hãy viết lại mô tả gốc thành 2 phiên bản hoàn toàn khác nhau về từ vựng và cấu trúc câu, nhưng phải tuân thủ TUYỆT ĐỐI các quy tắc sau:
    1. KHÔNG được nhắc đến bất kỳ tên siêu thị nào (như VinMart, Bách hóa XANH, Co.opmart). Chỉ dùng từ "AgriTech" hoặc "chúng tôi".
    2. KHÔNG được tự ý thêm các món ăn phi logic (ví dụ: cấm xào thịt gà với gan lợn).
    3. Trả về kết quả ĐÚNG định dạng sau, không giải thích gì thêm:
    [Phiên bản 1]
    (Nội dung phiên bản 1 ở đây)
    [Phiên bản 2]
    (Nội dung phiên bản 2 ở đây)
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Lỗi khi gọi API cho sản phẩm {ten_sp}: {e}")
        return None

def run_augmentation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "nong_san_ready_for_ai.csv")
    
    if not os.path.exists(input_file):
        print(f"Không tìm thấy file {input_file}.")
        return

    df = pd.read_csv(input_file)
    print(f"Bắt đầu nhân bản {len(df)} mẫu dữ liệu...\n")

    augmented_records = []

    for index, row in df.iterrows():
        print(f"Đang xử lý [{index + 1}/{len(df)}]: {row['Ten_SP']}")
        
        augmented_records.append({
            "Ten_SP": row['Ten_SP'],
            "Danh_Muc": row['Danh_Muc'],
            "Mo_Ta": row['Mo_Ta']
        })

        response_text = augment_description(row['Ten_SP'], row['Danh_Muc'], row['Mo_Ta'])
        
        if response_text:
            if "[Phiên bản 1]" in response_text and "[Phiên bản 2]" in response_text:
                parts = response_text.split("[Phiên bản 2]")
                
                pb1 = parts[0].replace("[Phiên bản 1]", "").strip()
                pb2 = parts[1].strip()
                
                if len(pb1) > 100:
                    augmented_records.append({"Ten_SP": row['Ten_SP'], "Danh_Muc": row['Danh_Muc'], "Mo_Ta": pb1})
                if len(pb2) > 100:
                    augmented_records.append({"Ten_SP": row['Ten_SP'], "Danh_Muc": row['Danh_Muc'], "Mo_Ta": pb2})
        
        time.sleep(2)

    df_augmented = pd.DataFrame(augmented_records)
    output_file = os.path.join(current_dir, "nong_san_augmented.csv")
    df_augmented.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\nTuyệt vời! Dữ liệu đã tăng từ {len(df)} lên {len(df_augmented)} mẫu.")
    print(f"File mới được lưu tại: {output_file}")

if __name__ == "__main__":
    run_augmentation()