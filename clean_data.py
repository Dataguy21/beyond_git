import pandas as pd
import sys

input_file, output_file = sys.argv[1], sys.argv[2]
df = pd.read_csv(input_file)
df_cleaned = df.drop_duplicates().dropna()
df_cleaned.to_csv(output_file, index=False)