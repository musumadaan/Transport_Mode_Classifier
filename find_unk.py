from pathlib import Path
import pandas as pd

df = pd.read_csv("features_output.csv")
print("Label counts from features_output.csv:")
print(df['label'].value_counts())
print("\nFolders that were labeled 'unknown':")
print(df[df['label'] == 'unknown']['source_id'].unique())