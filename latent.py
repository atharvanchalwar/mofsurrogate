import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from vaemof.model import VAEMOF
from vaemof.configs import AttributeDict

# === Paths ===
results_dir = "results11/best"
data_path = "../data/MOF_properties_train.csv"
output_csv_path = "latent_space_with_properties_8.csv"
output_merged_path = "latent_space_full_8.csv"
batch_size = 256

# === Load model ===
config = AttributeDict.from_jsonfile(f"{results_dir}/config.json")
config.files_results = results_dir
model = VAEMOF.load(config)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# === Load and clean property data
df = pd.read_csv(data_path)

# Drop rows where selfies are not safe
if 'selfies_safe' in df.columns:
    df = df[df['selfies_safe'] == True]

# Drop unused columns
drop_cols = ['selfies_safe', 'mask', 'train/test']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Identify key columns
smiles_col = "branch_smiles"
composite_key_cols = ['branch_smiles', 'organic_core', 'metal_node', 'topology']
y_cols = [col for col in config['y_labels'] if col in df.columns]
non_mof_cols = composite_key_cols + y_cols
mof_candidates = [col for col in df.columns if col not in non_mof_cols]

# === Filter rows with unseen MOF vocab values
print("\n🧼 Filtering rows with unseen MOF values...")
initial_len = len(df)

if isinstance(model.vocab_mof.encoders, dict):
    for col, enc in model.vocab_mof.encoders.items():
        if isinstance(enc, LabelEncoder) and col in df.columns:
            before = len(df)
            df = df[df[col].isin(enc.classes_)]
            print(f"Dropped {before - len(df)} rows due to unseen values in '{col}'")
else:
    for col, enc in zip(mof_candidates, model.vocab_mof.encoders):
        if isinstance(enc, LabelEncoder):
            before = len(df)
            df = df[df[col].isin(enc.classes_)]
            print(f"Dropped {before - len(df)} rows due to unseen values in '{col}'")

print(f"✅ Final dataset size after MOF filtering: {len(df)} rows")

# === Convert to tuples
data_tuples = model.df_to_tuples(df, smiles_col)

# === Encode to latent space
all_z = []
for i in tqdm(range(0, len(data_tuples), batch_size), desc="Encoding"):
    batch = data_tuples[i:i + batch_size]
    tensors = model.tuples_to_tensors(batch)
    with torch.no_grad():
        z = model.inputs_to_z(tensors["x"], tensors.get("mof"))
        all_z.append(z.cpu().numpy())

# === Format latent space output
z_matrix = np.concatenate(all_z, axis=0)
z_df = pd.DataFrame(z_matrix, columns=[f"z_{i}" for i in range(z_matrix.shape[1])])

# Include composite key columns
z_df = pd.concat([df[composite_key_cols].reset_index(drop=True), z_df], axis=1)

# Include property targets
y_df = df[y_cols].reset_index(drop=True)
final_df = pd.concat([z_df, y_df], axis=1)

# === Save latent + targets
final_df.to_csv(output_csv_path, index=False)
print(f"\n✅ Latent + property matrix saved to: {output_csv_path}")

# === Merge full metadata using composite key
df_props_full = pd.read_csv(data_path).drop(columns=drop_cols, errors='ignore')
merged = final_df.merge(df_props_full, on=composite_key_cols, how='left')
merged.to_csv(output_merged_path, index=False)
print(f"✅ Full latent space + MOF metadata saved to: {output_merged_path}")
