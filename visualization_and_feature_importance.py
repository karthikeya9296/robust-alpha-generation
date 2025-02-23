import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

# 🔄 Recreate the LSTM Model Architecture
def create_model():
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50, return_sequences=False),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ✅ Load the model weights
model = create_model()
model.load_weights("model/lstm_model.h5")
print("✓ Model weights loaded successfully")

# 🔄 Load and scale the data
data = pd.read_csv('engineered_stock_data.csv', index_col='Date', parse_dates=True)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close_AAPL']].values)

# 🏗️ Prepare the input sequences
SEQUENCE_LENGTH = 60
X = []
for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i - SEQUENCE_LENGTH:i])

X = np.array(X)
X = X.reshape((X.shape[0], SEQUENCE_LENGTH, 1))

# 🟠 Use a subset of samples for SHAP analysis
n_samples = 200
background = X[:n_samples]
test_samples = X[:n_samples]

# ⚡ SHAP explainer using GradientExplainer
explainer = shap.GradientExplainer(model, background)

# 🔍 Compute SHAP values
shap_values = explainer.shap_values(test_samples)

# ✅ Process SHAP values
if isinstance(shap_values, list):
    shap_values_combined = np.sum([np.array(sv) for sv in shap_values], axis=0)
else:
    shap_values_combined = np.array(shap_values)

# 🔄 Reshape SHAP values for visualization
shap_values_flat = shap_values_combined.reshape((n_samples, -1))
test_samples_flat = test_samples.reshape((n_samples, -1))

# 🏷️ Feature names
feature_names = [f'TimeStep_{i}_Close_AAPL' for i in range(SEQUENCE_LENGTH)]

# 📊 Generate and save the summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_flat,
    test_samples_flat,
    feature_names=feature_names,
    plot_type="bar",
    show=False  # Don't show the plot yet
)

# 💾 Save the SHAP summary plot
plt.savefig("results/feature_importance.png", bbox_inches='tight', dpi=300)
plt.show()

# 📋 Create a detailed feature importance analysis
feature_importance = np.abs(shap_values_flat).mean(0)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# 💾 Save feature importance to CSV
importance_df.to_csv("results/feature_importance.csv", index=False)
print("\n✅ Feature importance analysis completed and saved!")

# 🔝 Print top 10 most important features
print("\nTop 10 most important time steps:")
print(importance_df.head(10))
