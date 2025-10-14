import pandas as pd
import numpy as np

file_path = r'C:\Users\Abhinav Nigade\OneDrive\Desktop\AmpRenew\.dist\ev_battery_charging_data.csv'  #change file path
df = pd.read_csv(file_path)
print(df.head(5))

#print("\nSummary statistics (numeric columns only):\n")
#summary = df.describe()
#print(summary)
df.replace("?", np.nan, inplace = True)

missing_data = df.isnull()
#print(missing_data.head(5))

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 