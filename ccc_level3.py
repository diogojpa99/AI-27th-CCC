import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = 'input/encrypted/train_data.csv'

train_data = pd.read_csv(file_path)

train_data = train_data.dropna()

# Find and remove rows with outliers in the 'amps' column
train_data = train_data[(train_data['AMPS'] > 0) & (train_data['AMPS'] < 1)]

# Get the number of rows remaining in the training data
num_rows = train_data.shape[0]

# Print the number of rows remaining
print("Number of rows remaining after removing problematic rows:", num_rows)

# convert Kelvin to Celsius
kelvin_mask = train_data['UNIT'] == 'K'
celsius_values = train_data.loc[kelvin_mask, 'TEMP'] - 273.15
train_data.loc[kelvin_mask, 'TEMP'] = celsius_values

#train_data.to_csv("cleaned_train_data.csv", index=False)

# select the rows with unknown unit
unknown_mask = train_data['UNIT'] == '?'
unknown_values = train_data.loc[unknown_mask, 'TEMP']

# plot Celsius and unknown values
fig, ax = plt.subplots()
ax.scatter(train_data['TEMP'], train_data['UNIT'])
ax.scatter(celsius_values, np.zeros(len(celsius_values)), color='r')
ax.scatter(unknown_values, np.ones(len(unknown_values)), color='k')
ax.set_xlabel('Temperature (Celsius)')
ax.set_ylabel('Unit')
ax.legend(['Data', 'Celsius', 'Unknown'])
plt.show()

# select rows with unknown unit and temperature > 150
unknown_mask = train_data['UNIT'] == '?'
high_temp_mask = train_data['TEMP'] > 150
selected_rows = train_data.loc[unknown_mask & high_temp_mask]

# convert selected values from Kelvin to Celsius
selected_values = selected_rows['TEMP'] - 273.15

# subtract 273.15 from selected values
train_data.loc[unknown_mask & high_temp_mask, 'TEMP'] = selected_values

# select the rows with unknown unit
unknown_mask = train_data['UNIT'] == '?'
unknown_values = train_data.loc[unknown_mask, 'TEMP']

# plot Celsius and unknown values
fig, ax = plt.subplots()
ax.scatter(train_data['TEMP'], train_data['UNIT'])
ax.scatter(celsius_values, np.zeros(len(celsius_values)), color='r')
ax.scatter(unknown_values, np.ones(len(unknown_values)), color='k')
ax.set_xlabel('Temperature (Celsius)')
ax.set_ylabel('Unit')
ax.legend(['Data', 'Celsius', 'Unknown'])
plt.show()


kelvin_mean = round(celsius_values.mean(), 2)
kelvin_std = round(celsius_values.std(), 0)

# print the mean and standard deviation of Kelvin values after conversion to Celsius
print(f"Mean: {kelvin_mean}\nStandard deviation: {kelvin_std}")