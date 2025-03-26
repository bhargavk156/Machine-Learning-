# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

# Sample data

data = pd.DataFrame({
    'size' :[1500,1800,2400,3000,3500],
    'age':[10,15,20,5,8],
    'price':[300000,350000,500000,600000,650000]
})

x = data[['size','age']]
y = data['price']

 # Linear Regression Model
model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

r2 = r2_score(y,y_pred)

n = x.shape[0]
k = x.shape[1]
adjusted_r2 = 1- (1-r2) * (n-1) / (n-k-1)

print(f"R2 : {r2}")
print(f"Adjusted R2:{adjusted_r2}")

 #  .....................................................................................................
 ## age ML learning

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

# Sample data
data = pd.DataFrame({
    'size': [1500, 1800, 2400, 3000, 3500],
    'age': [10, 15, 20, 5, 1],
    'price': [300000, 350000, 500000, 600000, 650000]
})

# Features and target
X = data[['size', 'age']]
y = data['price']

# Linear Regression Model
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

# Plot actual vs predicted prices
plt.scatter(data['size'], y, color='blue', label='Actual Price')  # Actual points
plt.scatter(data['size'], y_pred, color='red', label='Predicted Price')  # Predicted points
plt.plot(data['size'], y_pred, color='red', linestyle='--')  # Line connecting predicted points

plt.xlabel('Size (sq.ft)')
plt.ylabel('Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()

 #  .....................................................................................................
 ## year ML learning

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

# Sample data with construction year
data = pd.DataFrame({
    'size': [1500, 1800, 2400, 3000, 3500],
    'construction_year': [2013, 2008, 2003, 2018, 2015],
    'price': [300000, 350000, 500000, 600000, 650000]
})

# Current year
current_year = 2025
data['age'] = current_year - data['construction_year']

# Features and target
X = data[['size', 'age']]
y = data['price']

# Train the Linear Regression Model
model = LinearRegression().fit(X, y)

# Predict current prices (for comparison)
y_pred = model.predict(X)

# R2 Score
r2 = r2_score(y, y_pred)
print(f"R²: {r2}")

# Predict prices for a future year (2025) by adjusting 'age'
future_year = 2030
data['age_future'] = future_year - data['construction_year']

# For prediction, keep the column names same ('size' and 'age')
future_X = data[['size', 'age_future']].rename(columns={'age_future': 'age'})
future_pred = model.predict(future_X)

print(f"\nPredicted Prices for year {future_year}:")
print(future_pred)

# Plot Actual Price, Current Predicted Price, and Future Predicted Price
plt.scatter(data['size'], y, color='blue', label='Actual Price (2025)')
plt.scatter(data['size'], y_pred, color='green', label='Predicted Price (2025)')
plt.scatter(data['size'], future_pred, color='red', label=f'Predicted Price ({future_year})')
plt.plot(data['size'], future_pred, color='red', linestyle='--')

plt.xlabel('Size (sq.ft)')
plt.ylabel('Price')
plt.title('House Price Prediction Year-wise')
plt.legend()
plt.show()

 #  .....................................................................................................
 ## Deep learning
pip install tensorflow

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
print(tf.__version__)   # ✅ TensorFlow is working

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Sample data with construction year
data = pd.DataFrame({
    'size': [1500, 1800, 2400, 3000, 3500],
    'construction_year': [2013, 2008, 2003, 2018, 2015],
    'price': [300000, 350000, 500000, 600000, 650000]
})

# Current year
current_year = 2024
data['age'] = current_year - data['construction_year']

# Features and target
X = data[['size', 'age']]
y = data['price']

# Feature scaling (Important for Deep Learning)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Build the Deep Learning Model (Best Practice with Input Layer)
model = Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_scaled, y, epochs=500, verbose=0)

# Predict current prices
y_pred = model.predict(X_scaled).flatten()

# R2 Score
r2 = r2_score(y, y_pred)
print(f"R² (Training): {r2}")

# ✅ Predict future prices (2025) - Fix feature name issue
future_year = 2025
data['age_future'] = future_year - data['construction_year']

# Rename 'age_future' to 'age' to match model input feature names
future_X = data[['size', 'age_future']].rename(columns={'age_future': 'age'})
future_X_scaled = scaler.transform(future_X)
future_pred = model.predict(future_X_scaled).flatten()

print(f"\nPredicted Prices for year {future_year}:")
print(future_pred)

# ✅ Plot Actual Price, Current Predicted Price, and Future Predicted Price
plt.scatter(data['size'], y, color='blue', label='Actual Price (2024)')
plt.scatter(data['size'], y_pred, color='green', label='DL Predicted Price (2024)')
plt.scatter(data['size'], future_pred, color='red', label=f'DL Predicted Price ({future_year})')
plt.plot(data['size'], future_pred, color='red', linestyle='--')

plt.xlabel('Size (sq.ft)')
plt.ylabel('Price')
plt.title('Deep Learning House Price Prediction Year-wise')
plt.legend()
plt.show()