import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# --- Step 1: Scrape company info from PSX ---

def get_company_list():
    url = "https://dps.psx.com.pk/market-watch"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    rows = soup.find_all('tr')
    data = []

    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 3:
            a_tag = cols[0].find('a')
            symbol = a_tag.find('strong').text.strip() if a_tag else None
            href = f"https://dps.psx.com.pk{a_tag['href']}" if a_tag else None
            title = a_tag.get('data-title') if a_tag else None
            sector_id = cols[1].text.strip()
            listed_in = cols[2].text.strip()

            data.append({
                'symbol': symbol,
                'href': href,
                'name': title,
                'sector_id': sector_id,
                'listed_in': listed_in
            })

    df = pd.DataFrame(data)

    # Map sector IDs to sector names
    sector_mapping = {
        "0801": "AUTOMOBILE ASSEMBLER",
        "0802": "AUTOMOBILE PARTS & ACCESSORIES",
        "0803": "CABLE & ELECTRICAL GOODS",
        "0804": "CEMENT",
        "0805": "CHEMICAL",
        "0806": "CLOSE - END MUTUAL FUND",
        "0807": "COMMERCIAL BANKS",
        "0808": "ENGINEERING",
        "0809": "FERTILIZER",
        "0810": "FOOD & PERSONAL CARE PRODUCTS",
        "0811": "GLASS & CERAMICS",
        "0812": "INSURANCE",
        "0813": "INV. BANKS / INV. COS. / SECURITIES COS.",
        "0814": "JUTE",
        "0815": "LEASING COMPANIES",
        "0816": "LEATHER & TANNERIES",
        "0818": "MISCELLANEOUS",
        "0819": "MODARABAS",
        "0820": "OIL & GAS EXPLORATION COMPANIES",
        "0821": "OIL & GAS MARKETING COMPANIES",
        "0822": "PAPER, BOARD & PACKAGING",
        "0823": "PHARMACEUTICALS",
        "0824": "POWER GENERATION & DISTRIBUTION",
        "0825": "REFINERY",
        "0826": "SUGAR & ALLIED INDUSTRIES",
        "0827": "SYNTHETIC & RAYON",
        "0828": "TECHNOLOGY & COMMUNICATION",
        "0829": "TEXTILE COMPOSITE",
        "0830": "TEXTILE SPINNING",
        "0831": "TEXTILE WEAVING",
        "0832": "TOBACCO",
        "0833": "TRANSPORT",
        "0834": "VANASPATI & ALLIED INDUSTRIES",
        "0835": "WOOLLEN",
        "0836": "REAL ESTATE INVESTMENT TRUST",
        "0837": "EXCHANGE TRADED FUNDS",
        "0838": "PROPERTY"
    }

    df['sector_name'] = df['sector_id'].map(sector_mapping)

    # One-hot encode 'listed_in' column
    df['listed_in_split'] = df['listed_in'].str.split(',')
    one_hot = df.explode('listed_in_split')
    listed_in_dummies = pd.get_dummies(one_hot['listed_in_split']).groupby(one_hot.index).sum()
    df = df.drop(columns='listed_in_split').join(listed_in_dummies)

    # Add time series data-link
    df['data-link'] = "https://dps.psx.com.pk/timeseries/eod/" + df['symbol']

    # Save to CSV
    df.to_csv('company_data.csv', index=False)
    print(f"✅ Saved company data to 'company_data.csv' with {len(df)} companies.")

    return df


# --- Step 2: Fetch EOD timeseries for each company ---

def fetch_timeseries(df):
    all_5y_data = []

    for i, symbol in enumerate(df['symbol']):
        url = f"https://dps.psx.com.pk/timeseries/eod/{symbol}"
        print(f"[{i+1}/{len(df)}] Fetching {symbol} ...")

        try:
            response = requests.get(url)
            data = response.json()

            if data.get("status") == 1 and "data" in data:
                company_data = data["data"]

                company_df = pd.DataFrame(company_data, columns=["timestamp", "close_price", "volume", "open_price"])
                company_df["timestamp"] = pd.to_datetime(company_df["timestamp"], unit="s")
                company_df["symbol"] = symbol

                all_5y_data.append(company_df)
            else:
                print(f"⚠️ No data for {symbol}")

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")

        time.sleep(1)  # Avoid hammering the server

    if all_5y_data:
        combined_df = pd.concat(all_5y_data, ignore_index=True)
        combined_df.to_csv("all_companies_timeseries.csv", index=False)
        print(f"✅ Saved all timeseries to 'all_companies_timeseries.csv'")
    else:
        print("❌ No timeseries data fetched.")

    return all_5y_data


# --- Run all steps ---

if __name__ == "__main__":
    company_df = get_company_list()
    fetch_timeseries(company_df)
