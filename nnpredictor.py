import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# === Load data
print("📥 Loading data...")
df = pd.read_csv("latent_space_full_32.csv")
latent_cols = [f'z_{i}' for i in range(32)]
target_col = 'co2ch4_co2_mol_kg_y'
df = df[latent_cols + [target_col]].dropna()

# === Preprocess features
print("🧹 Imputing and scaling features...")
X_raw = df[latent_cols].values
y_true = df[target_col].values

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# === Define the SAME model architecture
class PretrainedLatentNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# === Load the trained model
print("📦 Loading trained model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PretrainedLatentNN().to(device)
model.load_state_dict(torch.load("latent32_nn_pretrained.pt", map_location=device))
model.eval()

# === Make predictions
print("🔮 Making predictions...")
with torch.no_grad():
    preds = model(X_tensor.to(device)).cpu().numpy().flatten()

# === Save predictions to CSV
df["predicted_co2ch4"] = preds
df.to_csv("latent32_nn_predictions.csv", index=False)
print("💾 Saved predictions to latent32_nn_predictions.csv")

# === Evaluate
mae = mean_absolute_error(y_true, preds)
percent_error = 100 * mae / np.mean(y_true)

print("\n📈 Evaluation Results:")
print(f"   MAE: {mae:.4f}")
print(f"   Mean % Error: {percent_error:.2f}%")
