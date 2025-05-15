import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler

input_file, output_file = sys.argv[1], sys.argv[2]
df = pd.read_csv(input_file)
features = df.drop(columns=["species"])  # Target column is 'species'
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
df_scaled["species"] = df["species"]
df_scaled.to_csv(output_file, index=False)