import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
csv_filename = "./data/Pate_walking.csv"
df = pd.read_csv(csv_filename)


#handle missing timesteps by resampling to 100ms
#df = df.set_index(pd.to_timedelta(df['Timestamp'], unit='ms'))
#df = df.resample('100ms').mean().interpolate('linear').round(0).astype(int)


# Remove the column you want (e.g., 'Sensor3')
#df.drop(columns=['Label'], inplace=True)

# Add new column and determine its value
df['Label'] = 'walking'

# Change the value of a column based on the condition
#df.loc[df['Timestamp'] < 731598, 'Label'] = 'idle'
#df.loc[(df['Timestamp'] >= 71825) & (df['Timestamp'] <= 74005), 'Label'] = 'trick'
#df.loc[df['Timestamp'] > 757382, 'Label'] = 'onheels'

# Save the modified DataFrame back to a CSV file
df.to_csv(csv_filename, index=False)


print(f"Script executed and file saved to {csv_filename}")