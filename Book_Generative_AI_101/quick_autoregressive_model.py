# (1) Import libraries
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from pandas.tseries.offsets import DateOffset
import numpy as np

# (2) Read the CSV file
file_path = 'Electric_Production.csv'
data = pd.read_csv(file_path)

data.head()

# (3) Parsing the data
dates = pd.to_datetime(data['DATE']).values

# (4) Split into training and test set
train_size = int(len(data) * 0.8) # 80% of the data
train, test = data[:train_size], data[train_size:]

# (5) Fit an AR model with statsmodels
model = AutoReg(train['IPG2211A2N'], lags=5)
model_fit = model.fit()

# (6) Make predictions on the test data
# Number of additional future time steps to forecast
n_steps = 10
predictions_into_future = model_fit.predict(start=len(train), end=len(train)+len(test)+n_steps-1, dynamic=False)
predictions_up_to_data = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# (7) Evaluating the model (rmse)
error = sqrt(mean_squared_error(test['IPG2211A2N'], predictions_up_to_data))
print('Test RMSE: %.3f' % error)

# (8) Plotting the results

# Extend the date range for the x-axis to include the additional future time steps
# Splitting the dates for train and test
train_dates, test_dates = dates[:train_size], dates[train_size:]

# Extend test_dates by n_steps (here, you need to manually add 10 future dates)
future_dates = pd.date_range(test_dates[-1], periods=n_steps+1, freq='M')[1:] # Adjust frequency as needed
full_test_dates = np.concatenate([test_dates, future_dates])

# Plotting the actual values
plt.plot(test_dates, test['IPG2211A2N'].values, label='actual')

# Plotting the predictions up to the current data
plt.plot(test_dates, predictions_up_to_data, color='red', label='predicted')

# Plotting the future predictions
plt.plot(full_test_dates, predictions_into_future, color='green', label='predicted future')

plt.legend()
plt.show()






