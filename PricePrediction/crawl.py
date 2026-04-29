from playwright.sync_api import sync_playwright
import pandas as pd
import re
from datetime import datetime
import os
import sys

def scrape_all_markets():
    targets = [
        {
            "category": "rau_cu",
            "url": "https://gia247.com/gia-rau-cu-qua-hom-nay/",
            "cols": {"name": 1, "hcm": 2, "hn": 3, "market": 4}
        },
        {
            "category": "thit",
            "url": "https://gia247.com/gia-cac-loai-thit-hom-nay/",
            "cols": {"name": 0, "hcm": 1, "hn": 2, "market": 3} 
        },
        {
            "category": "thuy_hai_san",
            "url": "https://gia247.com/gia-thuy-hai-san/",
            "cols": {"name": 1, "hcm": 2, "hn": 3, "market": 4} 
        }
    ]

    scraped_data = []
    today_date = datetime.now().strftime('%Y-%m-%d')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        for target in targets:
            page = context.new_page()
            print(f"[{today_date}] Đang xử lý: {target['category'].upper()}...")
            
            try:
                page.goto(target['url'], timeout=60000, wait_until='domcontentloaded')
                
                first_table = page.query_selector('table')
                
                if not first_table:
                    print(f"Không tìm thấy bảng nào tại {target['category']}")
                    page.close()
                    continue

                table_rows = first_table.query_selector_all('tr')
                count = 0
                
                for row in table_rows:
                    cells = row.query_selector_all('td')
                    col_map = target['cols']
                    
                    if len(cells) > col_map['name']:
                        product_name = cells[col_map['name']].inner_text().strip()
                        
                        if not product_name or product_name.lower() in ['sản phẩm', 'thực phẩm', 'tên', 'stt']:
                            continue
                            
                        if product_name.upper() in ['USD', 'EUR', 'GBP', 'JPY'] or product_name.replace('.', '').isnumeric():
                            continue

                        def get_text_safe(key):
                            if key in col_map and len(cells) > col_map[key]:
                                return cells[col_map[key]].inner_text().strip()
                            return "0"

                        def process_price(raw_string):
                            cleaned_str = raw_string.replace('đ', '').replace('.', '').replace(',', '').strip()
                            
                            if not cleaned_str: return 0
                            
                            if '–' in cleaned_str or '-' in cleaned_str:
                                parts = re.split(r'[–-]', cleaned_str)
                                try: return int((int(parts[0].strip()) + int(parts[1].strip())) / 2)
                                except ValueError: return 0
                            try: return int(cleaned_str)
                            except ValueError: return 0
                        
                        scraped_data.append({
                            "date": today_date,
                            "category": target['category'],
                            "product_name": product_name,
                            "price_hcm": process_price(get_text_safe('hcm')),
                            "price_hn": process_price(get_text_safe('hn')),
                            "market_price": process_price(get_text_safe('market'))
                        })
                        count += 1
                
                print(f"-> Đã bóc tách {count} hàng dữ liệu sạch.")
                page.close()

            except Exception as e:
                print(f"Lỗi: {e}")
                page.close()

        browser.close()
    
    if scraped_data:
        df = pd.DataFrame(scraped_data)
        output_filename = 'PricePrediction/market_prices.csv'
        
        file_exists = os.path.isfile(output_filename)
        df.to_csv(output_filename, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
        print(f"\n[XONG] Tổng cộng {len(scraped_data)} dòng đã được lưu.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    scrape_all_markets()