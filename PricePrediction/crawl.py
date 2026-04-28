from playwright.sync_api import sync_playwright
import pandas as pd
import re

def scrape_vegetable_prices():
    scraped_data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        print("Navigating to the website...")
        target_url = "https://gia247.com/gia-rau-cu-qua-hom-nay/"
        page.goto(target_url)
        
        page.wait_for_selector('table', timeout=10000)
        print("Target table located successfully!")
        
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
                    
                    if not cleaned_str:
                        return 0
                        
                    if '–' in cleaned_str or '-' in cleaned_str:
                        parts = re.split(r'[–-]', cleaned_str)
                        try:
                            low = int(parts[0].strip())
                            high = int(parts[1].strip())
                            return int((low + high) / 2) # Return the average
                        except ValueError:
                            return 0
                    else:
                        # If it's a single price
                        try:
                            return int(cleaned_str)
                        except ValueError:
                            return 0
                
                price_hcm_clean = process_price(price_hcm_raw)
                price_hn_clean = process_price(price_hn_raw)
                market_price_clean = process_price(market_price_raw)
                
                scraped_data.append({
                    "product_name": product_name,
                    "price_hcm": price_hcm_clean,
                    "price_hn": price_hn_clean,
                    "market_price": market_price_clean
                })
                print(f"Extracted: {product_name} | HCM: {price_hcm_clean}")

        browser.close()
    
    df = pd.DataFrame(scraped_data)
    output_filename = 'vegetable_prices.csv'
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\nSuccess! Data exported to {output_filename}")

if __name__ == "__main__":
    scrape_vegetable_prices()