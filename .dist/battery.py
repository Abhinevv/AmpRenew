import pandas as pd
import numpy as np

file_path = r'C:\Users\Abhinav Nigade\OneDrive\Desktop\AmpRenew\.dist\ev_battery_charging_data.csv'  #change file path
data = pd.read_csv(file_path)
print(data.head(5))
