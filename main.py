import pandas as pd
from predict_prophet import train_and_predict
from datetime import datetime

def run_interface():
    print("📈 PSX Prophet Forecaster - Batch Mode for All Companies")
    print("Processing all companies in the dataset...\n")

    # Read all unique company symbols from the CSV
    df = pd.read_csv('all_companies_timeseries.csv', usecols=['symbol'])
    symbols = df['symbol'].unique()

    forecast_periods = {
        "1 day": 1,
        "1 week": 7,
        "1 month": 30,
        "6 months": 180
    }

    results = []
    for symbol in symbols:
        print(f"\n=== {symbol} ===")
        try:
            for label, days in forecast_periods.items():
                result = train_and_predict(symbol, days)
                print(f"\n📊 {label.upper()}:")
                print(f"📅 Date: {result['date_predicted']}")
                print(f"🔸 Predicted Price: Rs. {result['predicted_price']}")
                print(f"📈 Change: {result['percent_change']}% from Rs. {result['last_price']}")
                # Collect results for CSV
                results.append({
                    'symbol': symbol,
                    'period': label,
                    'date_predicted': result['date_predicted'],
                    'predicted_price': result['predicted_price'],
                    'percent_change': result['percent_change'],
                    'absolute_difference': result['absolute_difference'],
                    'last_price': result['last_price']
                })
        except Exception as e:
            print(f"⚠️ Error for {symbol}: {e}")
        print("\n-----------------------------\n")
    # Save results to CSV with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_filename = f'all_companies_forecast_results_{timestamp}.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}\n")

if __name__ == "__main__":
    run_interface()
