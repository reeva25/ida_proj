import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_excel('IDA STOCK 1.xlsx')
x = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=False)

model = LinearRegression()
model.fit(x_train, y_train)

print(model.score(x_test, y_test) * 100)

# Make predictions on the testing set
y_pred = model.predict(x_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)

# Convert 'Date' to datetime
dates = pd.to_datetime(df['Date'])
dates_train, dates_test = train_test_split(dates, test_size=0.25, shuffle=False)

# Plotting the results
plt.plot(dates_test, y_test, label='Actual Close Price')
plt.plot(dates_test, y_pred, color="pink", label='Predicted Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Price Over Time')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()


# plt.scatter(dates_test,y_test,marker='*')

