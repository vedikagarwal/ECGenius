import pandas as pd
from sklearn.ensemble import IsolationForest

file_path = "heart_failure_clinical_records_dataset.csv" 
df = pd.read_csv(file_path)


iso_forest = IsolationForest(contamination=0.05, random_state=42)

outliers = iso_forest.fit_predict(df)


df_cleaned = df[outliers == 1]

cleaned_file_path = 'cleaned_file.csv'
df_cleaned.to_csv(cleaned_file_path, index=False)

print("Outliers removed. Cleaned data saved to:", cleaned_file_path)
