import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import optuna

# === Load and preprocess data
print("📥 Loading data...")
df = pd.read_csv("latent_space_full_64.csv")
latent_cols = [f'z_{i}' for i in range(64)]
target_col = 'co2n2_co2_mol_kg_y'
df = df[latent_cols + [target_col]].dropna()

X_raw = df[latent_cols].values
y_raw = df[target_col].values

print("🧼 Imputing and scaling features...")
X = SimpleImputer().fit_transform(X_raw)
X = StandardScaler().fit_transform(X)
y = y_raw

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# === Define the NN class
class LatentNN(nn.Module):
    def __init__(self, input_dim, layers, dropout, activation):
        super().__init__()
        act_fn = nn.ReLU if activation == "relu" else nn.LeakyReLU
        units = []
        in_dim = input_dim
        for out_dim in layers:
            units.append(nn.Linear(in_dim, out_dim))
            units.append(nn.LayerNorm(out_dim))
            units.append(act_fn())
            units.append(nn.Dropout(dropout))
            in_dim = out_dim
        units.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*units)

    def forward(self, x):
        return self.net(x)

# === Optuna objective
def objective(trial):
    trial_number = trial.number
    print(f"\n🔁 Starting trial {trial_number}")

    # Trial params
    num_layers = trial.suggest_int("num_layers", 2, 4)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "leakyrelu"])
    layers = [hidden_size] * num_layers

    print(f"🧪 Trial {trial_number} params: layers={layers}, dropout={dropout:.2f}, lr={lr:.5f}, wd={weight_decay:.1e}, act={activation}")

    model = LatentNN(input_dim=64, layers=layers, dropout=dropout, activation=activation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == 49:
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    pred = model(xb).cpu().numpy().flatten()
                    val_preds.extend(pred)
                    val_targets.extend(yb.numpy().flatten())
            val_mae = mean_absolute_error(val_targets, val_preds)
            print(f"📅 Trial {trial_number} | Epoch {epoch:02d} | Val MAE: {val_mae:.4f}")

    final_mae = mean_absolute_error(val_targets, val_preds)
    print(f"✅ Trial {trial_number} complete. Final Val MAE: {final_mae:.4f}")
    return final_mae


# === Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# === Report
print("\n🏆 Best trial:")
best = study.best_trial
print(f"   MAE: {best.value:.4f}")
print(f"   Params: {best.params}")
