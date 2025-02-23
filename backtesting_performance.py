import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 📥 Load the trained LSTM model
model = load_model("model/lstm_model.h5")

# 📊 Load the engineered dataset
data = pd.read_csv("engineered_stock_data.csv", index_col="Date", parse_dates=True)

# ✅ Prepare the input data using only 'Close_AAPL'
features = data[['Close_AAPL']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# 🛠️ Prepare the data for backtesting
SEQUENCE_LENGTH = 60
X = []
for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i - SEQUENCE_LENGTH:i, 0])

X = np.array(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape to (samples, timesteps, features)

# 🔮 Make predictions using the trained model
predictions = model.predict(X)

# 🔄 Inverse transform predictions and actual data to get the original scale
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(scaled_data[SEQUENCE_LENGTH:])

# 📉 Evaluate the model performance
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"📊 Backtesting RMSE: {rmse}")

# 📈 Plot predictions vs actual prices
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, color='blue', label='Actual AAPL Price')
plt.plot(predicted_prices, color='red', label='Predicted AAPL Price')
plt.title('AAPL Price Backtesting using LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig("results/backtesting_predictions.png")
plt.show()

print("✅ Backtesting completed successfully!")
