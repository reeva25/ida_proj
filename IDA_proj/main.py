import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the stock price data
df = pd.read_excel('IDA STOCK 1.xlsx')
# Prepare the data
x = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
print(y)
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
model=LinearRegression()
model.fit(x_train,y_train)
print(model.score(x_test,y_test)*100)
# Make predictions on the testing set
y_pred = model.predict(x_test)
# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)
x_axis=df['Date']
plt.scatter(y_test,y_pred)
# Create the Random Forest model
# model = RandomForestRegressor()
# # Train the model
# model.fit(x_train, y_train)
#
# print(model.score(x_test,y_test)*100)
# # Make predictions on the testing set
# y_pred = model.predict(x_test)
#
# # Evaluate the model performance
# mse = mean_squared_error(y_test, y_pred)
# print('Mean squared error:', mse)