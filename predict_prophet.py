def train_and_predict(symbol, horizon_days, start_date=None):
    import pandas as pd
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    import plotly.graph_objects as go
    import logging
    from sklearn.metrics import r2_score, mean_squared_error

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    df = pd.read_csv('all_companies_timeseries.csv', parse_dates=['timestamp'])
    df = df[df['symbol'] == symbol].sort_values('timestamp')

    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df['timestamp'] <= start_date]
        if df.empty:
            raise ValueError("No data available before the given date.")

    prophet_df = df[['timestamp', 'close_price']].rename(columns={'timestamp': 'ds', 'close_price': 'y'})
    model = Prophet(yearly_seasonality=True)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

    # Plot full interactive forecast
    # fig = plot_plotly(model, forecast)
    # fig.update_layout(title=f"{symbol} - Forecast", xaxis_title="Date", yaxis_title="Price")
    # fig.show()
    #
    # Comparison: actual vs predicted for overlap
    comparison_df = pd.merge(
        prophet_df[['ds', 'y']],
        forecast[['ds', 'yhat']],
        on='ds',
        how='inner'
    )
    comparison_df['error'] = comparison_df['yhat'] - comparison_df['y']
    comparison_df['percent_error'] = (comparison_df['error'] / comparison_df['y']) * 100

    # # Plot actual vs predicted
    # fig2 = go.Figure()
    # fig2.add_trace(go.Scatter(x=comparison_df['ds'], y=comparison_df['y'], mode='lines+markers', name='Actual'))
    # fig2.add_trace(go.Scatter(x=comparison_df['ds'], y=comparison_df['yhat'], mode='lines+markers', name='Predicted'))
    # fig2.update_layout(
    #     title=f"{symbol} - Actual vs Predicted (Overlap)",
    #     xaxis_title="Date",
    #     yaxis_title="Price"
    # )
    # fig2.show()

    # Calculate and print R2 and MSE
    r2 = r2_score(comparison_df['y'], comparison_df['yhat'])
    mse = mean_squared_error(comparison_df['y'], comparison_df['yhat'])
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")



    # Find last known price and target date
    last_known_price = prophet_df.iloc[-1]['y']
    target_date = forecast.iloc[-1]['ds']
    predicted_price = forecast.iloc[-1]['yhat']
    percent_change = ((predicted_price - last_known_price) / last_known_price) * 100
    absolute_diff = predicted_price - last_known_price

    return {
        'predicted_price': round(predicted_price, 2),
        'percent_change': round(percent_change, 2),
        'absolute_difference': round(absolute_diff, 2),
        'last_price': round(last_known_price, 2),
        'date_predicted': target_date.strftime('%Y-%B-%d')
    }
