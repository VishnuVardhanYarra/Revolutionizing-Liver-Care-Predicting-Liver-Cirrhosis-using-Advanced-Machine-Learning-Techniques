import pandas as pd
df = pd.read_csv("Data/liver_data.csv")

# replace with the exact target column name you used
TARGET_COL = "Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)"

print("\nUnique raw labels and counts:")
print(df[TARGET_COL].value_counts(dropna=False))
