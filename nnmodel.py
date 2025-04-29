import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# === Load and preprocess data
print("📥 Loading data...")
df = pd.read_csv("latent_space_full_32.csv")
latent_cols = [f'z_{i}' for i in range(32)]
target_col = 'co2ch4_co2_mol_kg_y'

df = df[latent_cols + [target_col]].dropna()

X_raw = df[latent_cols].values
y_raw = df[target_col].values

print("🧼 Imputing and scaling...")
X = SimpleImputer(strategy='mean').fit_transform(X_raw)
X = StandardScaler().fit_transform(X)
y = y_raw

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# === Define Deep NN
class PretrainedLatentNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# === Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PretrainedLatentNN().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-5)
criterion = nn.MSELoss()

# === Training loop
epochs = 1000
print("🧠 Pretraining Deep NN on full dataset...")
for epoch in range(epochs):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0 or epoch == epochs - 1:
        model.eval()
        with torch.no_grad():
            preds = model(X_tensor.to(device)).cpu().numpy().flatten()
            targets = y
            mae = mean_absolute_error(targets, preds)
            r2 = r2_score(targets, preds)
            pct_err = 100 * mae / np.mean(targets)
            print(f"📅 Epoch {epoch:03d} | R²: {r2:.4f} | MAE: {mae:.4f} | % Error: {pct_err:.2f}%")

print("✅ Pretraining complete.")

# === Save model
print("💾 Saving model...")
torch.save(model.state_dict(), "latent32_nn_pretrained.pt")
print("✅ Model saved: latent64_nn_pretrained.pt")
