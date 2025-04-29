import pandas as pd
import numpy as np
import time
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import json

print("📥 Loading data...")
df = pd.read_csv("latent_space_full_32.csv")

# === Keep only latent descriptors and target
latent_cols = [f'z_{i}' for i in range(32)]
target_col = 'co2ch4_co2_mol_kg_y'
df = df[latent_cols + [target_col]]
df = df.dropna(subset=[target_col])
print(f"✅ Data loaded and filtered: {len(df)} samples.")

# === Feature and target arrays
print("🔍 Preprocessing features and target...")
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df[latent_cols])
y = df[target_col].values
print(f"✅ Preprocessing complete. Shape of X: {X.shape}, Shape of y: {y.shape}")

# === Define base models
print("🧠 Defining base models...")
models = {
    'rf': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    'xg': xgb.XGBRegressor(n_estimators=100, n_jobs=-1, tree_method='hist', random_state=42, verbosity=0),
    'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# === Train base models on full data
r2_scores = {}
for name, model in models.items():
    print(f"⚙️  Training {name}...")
    start = time.time()
    model.fit(X, y)
    r2_scores[name] = model.score(X, y)
    print(f"   ✅ {name} — R²: {r2_scores[name]:.4f}, Time: {time.time() - start:.2f}s")

# === Compute ensemble weights
total = sum(r2_scores.values())
weights = [r2_scores[name] / total for name in models]
weight_display = {name: round(w, 4) for name, w in zip(models, weights)}
print(f"📊 Ensemble Weights: {weight_display}")

# === Build VotingRegressor
print("🧩 Building ensemble VotingRegressor...")
voting_model = VotingRegressor(
    estimators=[(name, models[name]) for name in models],
    weights=weights,
    n_jobs=-1
)
voting_model.fit(X, y)
print("✅ Ensemble trained on entire dataset.")

# === Save everything
print("💾 Saving model and components...")
joblib.dump(voting_model, "latent32_ensemble_model.pkl")
joblib.dump(imputer, "latent32_imputer.pkl")
with open("latent32_ensemble_weights.json", "w") as f:
    json.dump(weight_display, f, indent=2)

print("Done")