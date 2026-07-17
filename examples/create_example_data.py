import pandas as pd
import numpy as np

np.random.seed(42)
n = 200

data = {
    'Age_Years': np.random.normal(55, 12, n).round(1),
    'Gender': np.random.choice(['Male', 'Female'], n),
    'BMI': np.random.normal(28, 5, n).round(1),
    'HbA1c': np.random.normal(7.5, 1.5, n).round(1),
    'Diabetes_Duration': np.random.exponential(8, n).round(1),
    'SBP': np.random.normal(130, 15, n).round(1),
    'DBP': np.random.normal(80, 10, n).round(1),
    'Smoking': np.random.choice(['Never', 'Former', 'Current'], n, p=[0.5, 0.3, 0.2]),
    'Hypertension': np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6]),
    'Nephropathy': np.random.choice(['Yes', 'No'], n, p=[0.35, 0.65]),
    'Cholesterol': np.random.normal(190, 40, n).round(1),
    'Retinopathy': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
}

df = pd.DataFrame(data)

df.loc[df['Retinopathy'] == 'Yes', 'HbA1c'] += np.random.normal(1, 0.5, df['Retinopathy'].value_counts()['Yes'])
df.loc[df['Retinopathy'] == 'Yes', 'Diabetes_Duration'] += np.random.normal(3, 1, df['Retinopathy'].value_counts()['Yes'])
df.loc[df['Retinopathy'] == 'Yes', 'BMI'] += np.random.normal(1, 0.5, df['Retinopathy'].value_counts()['Yes'])
df['HbA1c'] = df['HbA1c'].clip(4, 14).round(1)
df['Diabetes_Duration'] = df['Diabetes_Duration'].clip(0, 30).round(1)
df['BMI'] = df['BMI'].clip(15, 50).round(1)

df.to_csv('example_medical_data.csv', index=False)
print(f"Created example_medical_data.csv with {len(df)} rows and {len(df.columns)} columns")
print("Columns:", list(df.columns))
