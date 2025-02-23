import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 📥 Load Data and Model
data = pd.read_csv('engineered_stock_data.csv', index_col='Date', parse_dates=True)
model = load_model('model/tuned_lstm_model.h5')

# 📊 Prepare Data for Prediction
features = data[['Close_AAPL', 'High_AAPL', 'Low_AAPL', 'Open_AAPL', 'Volume_AAPL', 'sentiment']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# Function to create sequences for LSTM
def create_sequences(data, sequence_length):
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
    return np.array(X)

# Create sequences
SEQUENCE_LENGTH = 60
X = create_sequences(scaled_data, SEQUENCE_LENGTH)

# 📈 Predict Prices
predicted_prices = model.predict(X)
predicted_prices_rescaled = scaler.inverse_transform(
    np.hstack((predicted_prices, np.zeros((len(predicted_prices), scaled_data.shape[1] - 1))))
)[:, 0]

# 📉 Generate Buy/Sell Signals
signals = []
for i in range(len(predicted_prices_rescaled) - 1):
    if predicted_prices_rescaled[i+1] > predicted_prices_rescaled[i]:
        signals.append(1)  # Buy
    else:
        signals.append(-1)  # Sell
signals.append(0)  # No signal for the last prediction

# 💰 Simulate Trades
capital = 100000  # Starting capital
positions = []  # Track positions
cash = capital
holdings = 0  # Number of shares held
trades = []  # Trade history

actual_prices = data['Close_AAPL'].values[SEQUENCE_LENGTH:]

for i in range(len(signals)):
    price = actual_prices[i]
    if signals[i] == 1 and cash >= price:
        shares_to_buy = cash // price
        holdings += shares_to_buy
        cash -= shares_to_buy * price
        trades.append(('Buy', i, price, holdings))
    elif signals[i] == -1 and holdings > 0:
        cash += holdings * price
        trades.append(('Sell', i, price, holdings))
        holdings = 0

# Final portfolio value
final_portfolio_value = cash + holdings * actual_prices[-1]
cumulative_returns = (final_portfolio_value - capital) / capital

# 📊 Performance Metrics
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    return (np.mean(returns) - risk_free_rate) / np.std(returns)

returns = np.diff(actual_prices) / actual_prices[:-1]
sharpe_ratio = calculate_sharpe_ratio(returns)

# 📈 Plot the Strategy (with better clarity)
plt.figure(figsize=(16, 8))
plt.plot(actual_prices, color='blue', alpha=0.7, label='Actual AAPL Price')
plt.plot(predicted_prices_rescaled, color='red', alpha=0.6, label='Predicted AAPL Price')

# Mark Buy and Sell signals clearly
for trade in trades:
    action, index, price, shares = trade
    if action == 'Buy':
        plt.scatter(index, price, marker='^', color='green', label='Buy Signal' if trades.index(trade) == 0 else "", s=50)
    elif action == 'Sell':
        plt.scatter(index, price, marker='v', color='red', label='Sell Signal' if trades.index(trade) == 0 else "", s=50)

plt.title('AAPL Trading Strategy Simulation')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='best')
plt.grid(True)

# Save and display the plot
plt.savefig('results/trading_strategy.png')
plt.show()  # Displays the plot window
plt.close()

# 📊 Display Performance
print(f"✅ Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"📈 Cumulative Returns: {cumulative_returns:.2%}")
print(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}")
print("📊 Trading strategy plot saved as 'results/trading_strategy.png'")
