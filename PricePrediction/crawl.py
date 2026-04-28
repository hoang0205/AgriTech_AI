from playwright.sync_api import sync_playwright
import pandas as pd
import re
from datetime import datetime
import os
import sys

def scrape_vegetable_prices():
    scraped_data = []
    
    today_date = datetime.now().strftime('%Y-%m-%d')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        target_url = "https://gia247.com/gia-rau-cu-qua-hom-nay/"
        print(f"[{today_date}] Đang truy cập trang web...")
        
        try:
            page.goto(target_url, timeout=60000, wait_until='domcontentloaded')
            print("Đã tải xong cấu trúc web!")
        except Exception as e:
            print(f"Lỗi truy cập mạng hoặc web bị chặn: {e}")
            browser.close()
            sys.exit(1) 
        
        try:
            page.wait_for_selector('table', timeout=15000)
        except Exception:
            print("Không tìm thấy bảng dữ liệu trên web. Có thể cấu trúc web đã đổi.")
            browser.close()
            sys.exit(1)

        table_rows = page.query_selector_all('tr')
        
        for row in table_rows:
            cells = row.query_selector_all('td')
            if len(cells) >= 5:
                product_name = cells[1].inner_text().strip()
                price_hcm_raw = cells[2].inner_text().strip()
                price_hn_raw = cells[3].inner_text().strip()
                market_price_raw = cells[4].inner_text().strip()

                def process_price(raw_string):
                    cleaned_str = raw_string.replace('đ', '').replace('.', '').strip()
                    if not cleaned_str: return 0
                    if '–' in cleaned_str or '-' in cleaned_str:
                        parts = re.split(r'[–-]', cleaned_str)
                        try: return int((int(parts[0].strip()) + int(parts[1].strip())) / 2)
                        except ValueError: return 0
                    else:
                        try: return int(cleaned_str)
                        except ValueError: return 0
                
                # Gom dữ liệu
                scraped_data.append({
                    "date": today_date,
                    "product_name": product_name,
                    "price_hcm": process_price(price_hcm_raw),
                    "price_hn": process_price(price_hn_raw),
                    "market_price": process_price(market_price_raw)
                })

        browser.close()
    
    if not scraped_data:
        print("Cảnh báo: Không bóc tách được dòng dữ liệu nào!")
        sys.exit(1)

    df = pd.DataFrame(scraped_data)
    
    output_filename = 'PricePrediction/vegetable_prices.csv'
    
    file_exists = os.path.isfile(output_filename)
    df.to_csv(output_filename, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
    
    print(f"\nThành công! Đã thêm {len(scraped_data)} mặt hàng vào file {output_filename}")

if __name__ == "__main__":
    scrape_vegetable_prices()