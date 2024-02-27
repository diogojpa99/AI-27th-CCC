import numpy as np
import pandas as pd

# load the dataset
df = pd.read_csv('input/encrypted/train_data.csv')

df = df.dropna()

# Find and remove rows with outliers in the 'amps' column
df = df[(df['AMPS'] >= 0) & (df['AMPS'] <= 1)]

# Get the number of rows remaining in the training data
num_rows = df.shape[0]

# Print the number of rows remaining
print("Number of rows remaining after removing problematic rows:", num_rows)

# convert Kelvin to Celsius
kelvin_mask = df['UNIT'] == 'K'
print(len(kelvin_mask))
k_c_values = df.loc[kelvin_mask, 'TEMP'] - 273.15
print(len(k_c_values))
df.loc[kelvin_mask, 'TEMP'] = k_c_values

df = df[df['UNIT'] != '?']

temp_mean = round(k_c_values.mean(),2)
temp_std = k_c_values.std()

# print the mean and standard deviation of Kelvin values after conversion to Celsius
print(f"Mean: {temp_mean}\nStandard deviation: {temp_std}")