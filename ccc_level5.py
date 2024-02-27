import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.neural_network import MLPRegressor


# load the dataset
df = pd.read_csv('input/encrypted/train_data.csv')

df_test= pd.read_csv('input/encrypted/test_data.csv')


############################ Data preparaton ############################

df = df.dropna()

# Find and remove rows with outliers in the 'amps' column
df = df[(df['AMPS'] >= 0) & (df['AMPS'] <= 1)]

# Get the number of rows remaining in the training data
num_rows = df.shape[0]

# Print the number of rows remaining
#print("Number of rows remaining after removing problematic rows:", num_rows)

# convert Kelvin to Celsius
kelvin_mask = df['UNIT'] == 'K'
celsius_values = df.loc[kelvin_mask, 'TEMP'] - 273.15
df.loc[kelvin_mask, 'TEMP'] = celsius_values

# select rows with unknown unit and temperature > 150
unknown_mask = df['UNIT'] == '?'
high_temp_mask = df['TEMP'] > 150
selected_rows = df.loc[unknown_mask & high_temp_mask]

# convert selected values from Kelvin to Celsius
selected_values = selected_rows['TEMP'] - 273.15

# subtract 273.15 from selected values
df.loc[unknown_mask & high_temp_mask, 'TEMP'] = selected_values

# select the rows with unknown unit
unknown_mask = df['UNIT'] == '?'
unknown_values = df.loc[unknown_mask, 'TEMP']


# plot Celsius and unknown values
fig, ax = plt.subplots()
ax.scatter(df['TEMP'], df['UNIT'])
ax.scatter(celsius_values, np.zeros(len(celsius_values)), color='r')
ax.scatter(unknown_values, np.ones(len(unknown_values)), color='k')
ax.set_xlabel('Temperature (Celsius)')
ax.set_ylabel('Unit')
ax.legend(['Data', 'Celsius', 'Unknown'])
plt.show()


#############################################################################

#### Preparing for the machine learning model

# drop the 'C' column
df = df.drop('UNIT', axis=1)
df_test = df_test.drop('UNIT', axis=1)

##### One-hot encoding #####
# Perform one-hot encoding on the 'power' column
df = pd.get_dummies(df, columns= ['POWER', 'MODE'], drop_first=True)
df_test = pd.get_dummies(df_test, columns= ['POWER', 'MODE'], drop_first=True)


######

X = df.drop(columns=['OUTPUT'])
y = df['OUTPUT']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model on the training set
# model = LinearRegression()
# model = MLPRegressor(hidden_layer_sizes=(12, 12), learning_rate_init=0.001, random_state=42)
model = MLPRegressor(hidden_layer_sizes=(256, 124, 64), learning_rate_init=0.0015, random_state=42,
                     solver= 'adam', activation='relu', max_iter=3000)

#model.fit(X_train, y_train)
model.fit(X, y)
X_test = df_test


# Evaluate the performance of the model on the testing set
y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
# print(f"Mean squared error: {mse:.2f}")
# print(f"Mean absolute error: {mae:.2f}")
# print(f"R-squared score: {r2:.2f}")

# print(y_pred)
for i in y_pred:
    print(i)