import pandas as pd
import sys
from sklearn.feature_selection import SelectKBest, chi2

input_file, output_file = sys.argv[1], sys.argv[2]
df = pd.read_csv(input_file)
X = df.drop(columns=["species"])
y = df["species"]
selector = SelectKBest(score_func=chi2, k=2)  # Select top 2 features
X_selected = selector.fit_transform(X, y)
selected_cols = X.columns[selector.get_support()].tolist()
df_selected = pd.DataFrame(X_selected, columns=selected_cols)
df_selected["species"] = y
df_selected.to_csv(output_file, index=False)