import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# load data
df = pd.read_excel("dulieu.xlsx", engine='openpyxl')
df.columns = ['x1', 'x2', 'x3', 'x4', 'y']

# convert data type
df = df.apply(pd.to_numeric, errors='coerce')

# drop rows with missing values
df.dropna(inplace=True)

# convert y to integer
df['y'] = df['y'].astype(int)

# split data into train and test sets
train_size = int(len(df) * 0.7)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# fit linear regression model
regressor = LinearRegression()
regressor.fit(train.iloc[:, :-1], train.iloc[:, -1])

# make predictions on test set
predictions = regressor.predict(test.iloc[:, :-1])

# print predictions
print(predictions)

# fit ARIMA model on y
model = ARIMA(train['y'], order=(1, 0, 0))
model_fit = model.fit()

# make predictions on test set
arima_predictions = model_fit.forecast(steps=len(test))[0]

# print ARIMA predictions
print(arima_predictions)

# combine predictions
combined_predictions = np.array([int(round((p + a) / 2)) for p, a in zip(predictions, arima_predictions)])

# print combined predictions
print(combined_predictions)
