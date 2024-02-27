from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.neural_network import MLPRegressor

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


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


#############################################################################

#### Preparing for the machine learning model

# drop the 'C' column
df = df.drop('UNIT', axis=1)
df_test = df_test.drop('UNIT', axis=1)

##### One-hot encoding #####
# Perform one-hot encoding on the 'power' column
df = pd.get_dummies(df, columns= ['POWER', 'MODE'], drop_first=True)
df_test = pd.get_dummies(df_test, columns= ['POWER', 'MODE'], drop_first=True)



########################## Start of the model ############################

X = df.drop(columns=['OUTPUT'])
y = df['OUTPUT']


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(df_test)

# Create the MLP model
model = Sequential()
model.add(Dense(264, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(y[:, None].shape[1]))


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=15, restore_best_weights=True)]
history = model.fit(X_train, y, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
#y_pred = model.predict(X_test)
# y_pred = y_pred.reshape(-1)

# Calculate performance metrics
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("MSE:", mse)
# print("MAE:", mae)

predictions = model.predict(X_test)
# print(predictions)

with open("pred_10.txt", "w") as f:
    for pred in predictions:
        f.write(str(pred[0]) + "\n")
