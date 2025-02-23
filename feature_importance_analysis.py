import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd

# Load the trained LSTM model
model = load_model("model/lstm_model.h5")

# Load the dataset
data = pd.read_csv('engineered_stock_data.csv', index_col='Date', parse_dates=True)

# Select relevant features
features = data[['Close_AAPL']].values

# Prepare the data
SEQUENCE_LENGTH = 60
X = []
for i in range(SEQUENCE_LENGTH, len(features)):
    X.append(features[i - SEQUENCE_LENGTH:i])

X = np.array(X)
X = X.reshape((X.shape[0], SEQUENCE_LENGTH, 1))

# Use fewer samples for SHAP analysis
n_samples = 200
background = X[:n_samples]
test_samples = X[:n_samples]

# Create explainer
explainer = shap.GradientExplainer(model, background)

# Get SHAP values
shap_values = explainer.shap_values(test_samples)

# Process SHAP values
if isinstance(shap_values, list):
    shap_values_combined = np.sum([np.array(sv) for sv in shap_values], axis=0)
else:
    shap_values_combined = np.array(shap_values)

# Reshape for visualization
shap_values_flat = shap_values_combined.reshape((n_samples, -1))
test_samples_flat = test_samples.reshape((n_samples, -1))

# Create feature names
feature_names = [f'TimeStep_{i}_Close_AAPL' for i in range(SEQUENCE_LENGTH)]

# Create and save summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_flat,
    test_samples_flat,
    feature_names=feature_names,
    plot_type="bar",
    show=False  # Don't show the plot yet
)

# Save the current figure
plt.savefig("results/feature_importance.png", bbox_inches='tight', dpi=300)

# Now display the plot
plt.show()

# Create a detailed feature importance analysis
feature_importance = np.abs(shap_values_flat).mean(0)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Save feature importance to CSV
importance_df.to_csv("results/feature_importance.csv", index=False)
print("\n✅ Feature importance analysis completed and saved!")

# Print top 10 most important features
print("\nTop 10 most important time steps:")
print(importance_df.head(10))