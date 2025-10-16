import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


file_path = r'C:\Users\Abhinav Nigade\OneDrive\Desktop\AmpRenew\.dist\ev_battery_charging_data.csv'
df = pd.read_csv(file_path)
print(df.head(5))


df.replace("?", np.nan, inplace=True)


missing_data = df.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
    break  

# check for factors affecting battery degradation rate
sns.regplot(x="Efficiency (%)", y="Degradation Rate (%)", data=df)
plt.ylim(0,)
plt.show()
