from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from bs4 import BeautifulSoup
import time
import os
import pandas as pd

def scrape_multiple_categories():
    base_url = "https://www.bachhoaxanh.com"
    
    # Danh sách các danh mục và link cần cào
    danh_muc_can_cao = {
        "Rau": "https://www.bachhoaxanh.com/rau-sach",
        "Củ": "https://www.bachhoaxanh.com/cu",
        "Nấm": "https://www.bachhoaxanh.com/nam-tuoi",
        "Trái cây": "https://www.bachhoaxanh.com/trai-cay-tuoi-ngon",
        "Thịt heo": "https://www.bachhoaxanh.com/thit-heo",
        "Thịt bò": "https://www.bachhoaxanh.com/thit-bo",
        "Thịt gà": "https://www.bachhoaxanh.com/thit-ga",
        "Hải sản": "https://www.bachhoaxanh.com/ca-tom-muc-ech",
        "Trứng": "https://www.bachhoaxanh.com/trung-ga",
    }

    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    # options.add_argument("--headless=new") # Chạy ngầm để máy đỡ giật
    
    # Sử dụng EdgeChromiumDriverManager như cấu hình máy của bạn
    driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)
    
    all_products = []

    try:
        for ten_danh_muc, url in danh_muc_can_cao.items():
            print(f"\n=== ĐANG CÀO DANH MỤC: {ten_danh_muc} ===")
            driver.get(url)
            time.sleep(5)
            
            # --- LỚP 1: LẤY DANH SÁCH LINK SẢN PHẨM ---
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            items = soup.find_all('h3', class_='product_name')
            
            current_category_products = []
            for item in items:
                ten_sp = item.text.strip()
                a_tag = item.find_parent('a') 
                
                if ten_sp and a_tag and 'href' in a_tag.attrs:
                    link_sp = base_url + a_tag['href']
                    current_category_products.append({
                        "Ten_SP": ten_sp,
                        "Link": link_sp,
                        "Danh_Muc": ten_danh_muc
                    })

            print(f"-> Tìm thấy {len(current_category_products)} sản phẩm trong danh mục {ten_danh_muc}. Bắt đầu chui vào từng trang lấy mô tả!")
            
            # --- LỚP 2: CHUI VÀO TỪNG LINK LẤY MÔ TẢ ---
            for sp in current_category_products:
                print(f"Đang cào: {sp['Ten_SP']}...")
                driver.get(sp['Link'])
                time.sleep(3)  
                
                detail_soup = BeautifulSoup(driver.page_source, 'html.parser')
                paragraphs = detail_soup.find_all('p')

                valid_texts = []            
                for p in paragraphs:
                    text = p.text.strip()
                    
                    # Điều kiện lọc: dài hơn 40 ký tự và không chứa từ khóa rác
                    is_boilerplate = any(keyword in text.lower() for keyword in ["tồn kho", "giao hàng", "đổi trả", "liên hệ trước"])
                    
                    if len(text) > 40 and not is_boilerplate:
                        valid_texts.append(text)
                        
                if valid_texts:
                    # Gộp toàn bộ các đoạn văn bản hợp lệ
                    mo_ta_gop = "\n".join(valid_texts)
                    # Tiền xử lý dữ liệu cho AI: Thay thế thương hiệu đối thủ bằng tên hệ thống AgriTech
                    mo_ta_gop = mo_ta_gop.replace("Bách hóa XANH", "AgriTech").replace("tại Bách hóa Xanh", "")
                    sp['Mo_Ta'] = mo_ta_gop
                else:
                    sp['Mo_Ta'] = "Đang cập nhật..."
                
                all_products.append(sp)

    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        driver.quit()
        
    return all_products

if __name__ == "__main__":
    data = scrape_multiple_categories()
    if data:
        df = pd.DataFrame(data)
        
        # Làm sạch: Lọc bỏ ngay các dòng "Đang cập nhật..." để xuất ra file CSV sạch tinh tươm
        df_clean = df[df['Mo_Ta'].notna() & (df['Mo_Ta'] != "Đang cập nhật...")].copy()
        
        if 'Link' in df_clean.columns:
            df_clean = df_clean.drop(columns=['Link']) 
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "tong_hop_nong_san_clean.csv")
        
        df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nThành công! Đã cào và làm sạch {len(df_clean)} sản phẩm đa danh mục.")
        print(f"File lưu tại đúng thư mục code: {output_file}")
    else:
        print("Không có dữ liệu trả về.")