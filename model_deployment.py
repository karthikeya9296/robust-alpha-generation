from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

def create_model():
    """
    Recreate the LSTM model architecture
    """
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50, return_sequences=False),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

try:
    # Create model and load weights
    model = create_model()
    model.load_weights("model/lstm_model.h5")
    print("✓ Model weights loaded successfully")
except Exception as e:
    print(f"Error loading model weights: {str(e)}")
    raise

# Load and prepare scaler
try:
    data = pd.read_csv('engineered_stock_data.csv', index_col='Date', parse_dates=True)
    scaler = MinMaxScaler()
    scaler.fit(data[['Close_AAPL']].values)
    print("✓ Scaler prepared successfully")
except Exception as e:
    print(f"Error preparing scaler: {str(e)}")
    raise

def prepare_input_data(sequence):
    """
    Prepare input sequence for prediction
    """
    try:
        sequence = np.array(sequence, dtype=np.float32).reshape(-1, 1)
        scaled_sequence = scaler.transform(sequence)
        X = np.reshape(scaled_sequence, (1, len(scaled_sequence), 1))
        return X
    except Exception as e:
        raise ValueError(f"Error preparing input data: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_summary": str(model.summary())
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions using the loaded model
    """
    try:
        data = request.get_json(force=True)
        input_sequence = data.get('sequence', [])

        # Validate input
        if not isinstance(input_sequence, list):
            return jsonify({"error": "Input sequence must be a list of numbers"}), 400
            
        if len(input_sequence) != 60:
            return jsonify({
                "error": "Input sequence must be exactly 60 values long",
                "provided_length": len(input_sequence)
            }), 400

        try:
            input_sequence = [float(x) for x in input_sequence]
        except (ValueError, TypeError):
            return jsonify({"error": "All values in sequence must be numbers"}), 400

        # Prepare input and make prediction
        X = prepare_input_data(input_sequence)
        prediction = model.predict(X, verbose=0)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        return jsonify({
            "status": "success",
            "predicted_price": float(predicted_price),
            "input_sequence_length": len(input_sequence)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("\nStarting Flask API server...")
    print("Available endpoints:")
    print("  - GET  /health  : Check if service is running")
    print("  - POST /predict : Make predictions")
    print("\nModel architecture:")
    model.summary()
    
    # Changed port to 5001 to avoid conflict with AirPlay
    PORT = 5001
    print(f"\nStarting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=True)