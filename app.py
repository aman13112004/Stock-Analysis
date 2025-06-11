from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load and preprocess data once
df = pd.read_csv("Final-50-stocks.csv")
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values(by='DATE')
df.set_index('DATE', inplace=True)
returns = df.pct_change().dropna()
stock_list = df.columns.tolist()

def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches="tight")
    img.seek(0)
    return base64.b64encode(img.read()).decode("utf-8")

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    if request.method == "POST":
        stock_name = request.form["stock"].upper()
        if stock_name not in df.columns:
            result['error'] = f"Stock '{stock_name}' not found in dataset."
        else:
            result["stock"] = stock_name

            # Price Over Time
            fig1 = plt.figure(figsize=(10, 4))
            plt.plot(df.index, df[stock_name], label=stock_name)
            plt.title(f"{stock_name} Price Over Time")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.grid(True)
            plt.legend()
            result["price_plot"] = plot_to_base64(fig1)
            plt.close()

            # Return Distribution
            fig2 = plt.figure(figsize=(8, 4))
            returns[stock_name].hist(bins=50, color='skyblue')
            plt.title(f'{stock_name} Daily Return Distribution')
            plt.xlabel('Daily Return')
            plt.ylabel('Frequency')
            result["return_plot"] = plot_to_base64(fig2)
            plt.close()

            # Volatility
            result["volatility"] = returns[stock_name].std()

            # Moving Averages
            df[f'{stock_name}_MA20'] = df[stock_name].rolling(20).mean()
            df[f'{stock_name}_MA50'] = df[stock_name].rolling(50).mean()

            fig3 = plt.figure(figsize=(10, 4))
            plt.plot(df.index, df[stock_name], label='Price')
            plt.plot(df.index, df[f'{stock_name}_MA20'], label='20-Day MA')
            plt.plot(df.index, df[f'{stock_name}_MA50'], label='50-Day MA')
            plt.title(f'{stock_name} - 20 & 50 Day Moving Averages')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            result["ma_plot"] = plot_to_base64(fig3)
            plt.close()

            # Linear Regression Prediction
            stock_df = df[[stock_name]].copy()
            stock_df['Shifted'] = stock_df[stock_name].shift(1)
            stock_df = stock_df.dropna()

            X = stock_df[['Shifted']]
            y = stock_df[stock_name]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fig4 = plt.figure(figsize=(10, 4))
            plt.plot(y_test.index, y_test, label='Actual')
            plt.plot(y_test.index, y_pred, label='Predicted', color='red')
            plt.title(f'{stock_name} Price Prediction (1-Day Ahead)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            result["prediction_plot"] = plot_to_base64(fig4)
            plt.close()

            result["mse"] = mean_squared_error(y_test, y_pred)
            result["r2"] = r2_score(y_test, y_pred)

            # Predict Next 3 Days
            last_price = df[stock_name].iloc[-1]
            future_prices = []
            for _ in range(3):
                next_price = model.predict([[last_price]])[0]
                future_prices.append(next_price)
                last_price = next_price
            result["future_prices"] = future_prices

    return render_template("index.html", result=result, stocks=stock_list)

if __name__ == "__main__":
    app.run(debug=True)
