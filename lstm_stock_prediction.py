# 📦 Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os

# 📊 Prepare the dataset for LSTM
def prepare_data(df, feature_col='Close_AAPL'):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[[feature_col]])

    # Create sequences for LSTM
    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

    # Train-test split (80% training, 20% testing)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test, scaler

# 🔨 Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 🚀 Train the model
def train_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(
        X_train, y_train, epochs=50, batch_size=32,
        validation_data=(X_test, y_test), verbose=1
    )
    return history

# 📊 Evaluate model performance
def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    print(f"📉 Root Mean Squared Error (RMSE): {rmse}")
    return rmse

# 📈 Plot predictions vs actual
def plot_predictions(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(14, 5))
    plt.plot(actual_prices, color='blue', label='Actual AAPL Price')
    plt.plot(predictions, color='red', label='Predicted AAPL Price')
    plt.title('AAPL Price Prediction using LSTM')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig("results/predictions.png")
    plt.show()

# 🔥 Main Execution Block
if __name__ == "__main__":
    print("🚀 Starting LSTM Model Training...")

    # Load the dataset
    df = pd.read_csv('engineered_stock_data.csv', index_col='Date', parse_dates=True)
    print("✅ Loaded engineered data, shape:", df.shape)

    # Prepare data for training
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    print("✅ Data prepared for training...")

    # Build and train model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    history = train_model(model, X_train, y_train, X_test, y_test)
    print("✅ Model trained successfully!")

    # Evaluate and save results
    rmse = evaluate_model(model, X_test, y_test, scaler)
    print(f"📊 Final RMSE: {rmse}")

    # Save the trained model
    os.makedirs("model", exist_ok=True)
    model.save("model/lstm_model.h5")
    print("💾 Model saved successfully to model/lstm_model.h5")

    # Generate predictions and plots
    os.makedirs("results", exist_ok=True)
    plot_predictions(model, X_test, y_test, scaler)
    print("📈 Predictions plot saved in results/predictions.png")
