# BiLSTM-based TWS Downscaling Code with Visualizations (2002–2024)

import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.backends.backend_pdf as pdf_backend
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed for reproducibility
def set_seed(seed=20):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(20)

# Create output directory
os.makedirs("figures", exist_ok=True)

# Load data
all_data = pd.read_excel("All_Data.xlsx")
tws_data = pd.read_excel("TWS_JPL.xlsx")

# Fill missing values in predictors
for col in ['SMS', 'ET', 'GWSA']:
    all_data[col] = all_data[col].bfill().ffill()

# Convert month name to number and create datetime
month_map = {month: i+1 for i, month in enumerate([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'])}
tws_data['Month_Num'] = tws_data['Month'].map(month_map)
tws_data['Date'] = pd.to_datetime(dict(year=tws_data.Year, month=tws_data.Month_Num, day=1))

# Drop duplicate dates and interpolate monthly TWS to daily
tws_data = tws_data.sort_values('Date').drop_duplicates(subset='Date')
tws_data.set_index('Date', inplace=True)
tws_monthly = tws_data['TWS'].resample('D').interpolate(method='linear').reset_index()

# Merge with predictor dataset
merged = pd.merge(all_data, tws_monthly, on='Date', how='inner')

# Create lagged features
def create_lagged_features(df, cols, lags=7):
    for col in cols:
        for lag in range(1, lags + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df.dropna()

predictors = ['SMS', 'ET', 'rainfall', 'runoff', 'GWSA']
lagged = create_lagged_features(merged.copy(), predictors, lags=7)

# Scale and reshape data
X = lagged[[f'{var}_lag{i}' for var in predictors for i in range(1, 8)]].values
y = lagged['TWS'].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_seq = X_scaled.reshape(-1, 7, len(predictors))
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_scaled, test_size=0.2, shuffle=False)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# Define BiLSTM model
class BiLSTMDownscaler(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = BiLSTMDownscaler(input_size=len(predictors))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.HuberLoss()

# Train the model
train_losses = []
for epoch in range(30):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch+1}/30, Loss: {train_losses[-1]:.4f}")

# Predict on full dataset
model.eval()
X_all_t = torch.tensor(X_seq, dtype=torch.float32)
with torch.no_grad():
    y_all_pred_scaled = model(X_all_t).numpy()
    y_all_pred = scaler_y.inverse_transform(y_all_pred_scaled)

# Save downscaled TWSA for entire period
pred_df = pd.DataFrame({
    'Date': lagged['Date'].values,
    'Actual_TWSA': y.flatten(),
    'Predicted_TWSA': y_all_pred.flatten()
})
pred_df.to_csv("figures/Downscaled_TWSA_Daily_2002_2024_BiLSTM.csv", index=False)

# Feature Importance Bar Plot (Simulated)
feature_names = [f'{var}_lag{i}' for var in predictors for i in range(1, 8)]
simulated_importance = np.random.rand(len(feature_names))
sorted_idx = np.argsort(simulated_importance)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_names)), simulated_importance[sorted_idx])
plt.xticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx], rotation=90)
plt.title("Simulated Feature Importance for BiLSTM Downscaling")
plt.tight_layout()
plt.savefig("figures/simulated_feature_importance_BiLSTM.png", dpi=300)
plt.close()

# Predicted vs actual Plot
plt.figure(figsize=(14, 5))
plt.plot(pred_df['Date'], pred_df['Actual_TWSA'], label='Actual TWSA')
plt.plot(pred_df['Date'], pred_df['Predicted_TWSA'], label='Predicted TWSA', linestyle='--')
plt.xlabel("Date")
plt.ylabel("TWSA")
plt.title("Actual vs Predicted TWSA (2002–2024) [BiLSTM]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/temporal_plot_TWSA_BiLSTM.png", dpi=300)
plt.close()

# Residual Plot
residuals = pred_df['Actual_TWSA'] - pred_df['Predicted_TWSA']
plt.figure(figsize=(14, 5))
plt.plot(pred_df['Date'], residuals, color='gray')
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals Plot (Actual - Predicted TWSA) [BiLSTM]")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/residuals_plot_TWSA_BiLSTM.png", dpi=300)
plt.close()

# Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(pred_df['Actual_TWSA'], pred_df['Predicted_TWSA'], alpha=0.5, s=10)
plt.plot([pred_df['Actual_TWSA'].min(), pred_df['Actual_TWSA'].max()],
         [pred_df['Actual_TWSA'].min(), pred_df['Actual_TWSA'].max()],
         color='red', linestyle='--')
plt.xlabel("Actual TWSA")
plt.ylabel("Predicted TWSA")
plt.title("Scatter Plot: Actual vs Predicted TWSA [BiLSTM]")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/scatter_plot_TWSA_BiLSTM.png", dpi=300)
plt.close()

# Training Loss Plot
plt.figure(figsize=(8, 4))
plt.plot(train_losses, marker='o')
plt.title("Training Loss Curve [BiLSTM]")
plt.xlabel("Epoch")
plt.ylabel("Huber Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/training_loss_curve_BiLSTM.png", dpi=300)
plt.close()

# Evaluation Metrics
mae = mean_absolute_error(pred_df['Actual_TWSA'], pred_df['Predicted_TWSA'])
rmse = np.sqrt(mean_squared_error(pred_df['Actual_TWSA'], pred_df['Predicted_TWSA']))
r2 = r2_score(pred_df['Actual_TWSA'], pred_df['Predicted_TWSA'])
