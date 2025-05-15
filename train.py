import pandas as pd
import sys
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json

input_file, model_file, metrics_file = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(input_file)
X = df.drop(columns=["species"])
y = df["species"]
model = LogisticRegression(max_iter=200)
model.fit(X, y)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

# Save the model
joblib.dump(model, model_file)
# Save the accuracy metric
metrics = {"accuracy": accuracy}
with open(metrics_file, "w") as f:
    json.dump(metrics, f)