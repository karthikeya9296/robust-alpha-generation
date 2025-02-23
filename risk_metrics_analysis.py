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

# 📉 Calculate Daily Returns
actual_prices = data['Close_AAPL'].values[SEQUENCE_LENGTH:]
returns = np.diff(actual_prices) / actual_prices[:-1]

# 📊 Risk Metrics
def value_at_risk(returns, confidence_level=0.05):
    return np.percentile(returns, 100 * confidence_level)

def conditional_value_at_risk(returns, confidence_level=0.05):
    var = value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()

def max_drawdown(prices):
    cumulative_returns = np.maximum.accumulate(prices)
    drawdowns = (prices - cumulative_returns) / cumulative_returns
    return drawdowns.min()

# Calculate Risk Metrics
VaR_95 = value_at_risk(returns, 0.05)
CVaR_95 = conditional_value_at_risk(returns, 0.05)
Max_Drawdown = max_drawdown(actual_prices)

# 📊 Display Metrics
print(f"📉 Value at Risk (95% Confidence): {VaR_95:.2%}")
print(f"📉 Conditional Value at Risk (CVaR) (95% Confidence): {CVaR_95:.2%}")
print(f"📉 Maximum Drawdown: {Max_Drawdown:.2%}")

# 📈 Plot Drawdowns
cumulative_returns = np.maximum.accumulate(actual_prices)
drawdowns = (actual_prices - cumulative_returns) / cumulative_returns

plt.figure(figsize=(16, 6))
plt.plot(drawdowns, color='red', label='Drawdown')
plt.title('AAPL Maximum Drawdown Analysis')
plt.xlabel('Time')
plt.ylabel('Drawdown (%)')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('results/drawdown_analysis.png')
plt.show(block=True)  # Ensure the plot stays until manually closed
plt.close()

print("📊 Risk metrics plot saved as 'results/drawdown_analysis.png'")
