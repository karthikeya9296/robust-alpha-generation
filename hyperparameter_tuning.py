import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("engineered_stock_data.csv", index_col="Date", parse_dates=True)
features = data[['Close_AAPL', 'High_AAPL', 'Low_AAPL', 'Open_AAPL', 'Volume_AAPL', 'sentiment']]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# Create sequences
def create_sequences(data, sequence_length=60):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 0])  # Close price
    return np.array(X), np.array(y)

# Prepare data
SEQUENCE_LENGTH = 60
X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Hyperparameter Tuning Function
def build_model(hp):
    model = Sequential()
    model.add(
        LSTM(
            units=hp.Int('units', min_value=50, max_value=300, step=50),
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    model.add(
        LSTM(
            units=hp.Int('units_2', min_value=50, max_value=200, step=50),
            return_sequences=False
        )
    )
    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
    model.add(Dense(1))

    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Tuner Setup
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='tuning_results',
    project_name='lstm_stock_tuning'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate model
predictions = best_model.predict(X_test)
predicted_prices = scaler.inverse_transform(
    np.hstack((predictions, np.zeros((len(predictions), scaled_data.shape[1] - 1))))
)
actual_prices = scaler.inverse_transform(
    np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), scaled_data.shape[1] - 1))))
)

# Plot predictions vs actual prices
plt.figure(figsize=(14, 5))
plt.plot(actual_prices[:, 0], color='blue', label='Actual AAPL Price')
plt.plot(predicted_prices[:, 0], color='red', label='Predicted AAPL Price')
plt.title('AAPL Price Prediction with Tuned LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('results/tuned_predictions.png')
plt.show()

# RMSE
rmse = np.sqrt(mean_squared_error(actual_prices[:, 0], predicted_prices[:, 0]))
print(f"🔍 Tuned Model RMSE: {rmse}")

# Save the tuned model
best_model.save("model/tuned_lstm_model.h5")
print("✅ Tuned model saved successfully!")
