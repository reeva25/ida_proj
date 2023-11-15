import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def Cleaning (file_path):
    df=pd.read_excel(file_path)
    mean_volume = df['Volume'].mean()
    print(f"Mean volume is {mean_volume}")

    q1 = df['Volume'].quantile(0.25)
    q3 = df['Volume'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    # # df=df.drop((df['Volume']>upper_bound)|(df['Volume']<lower_bound))
    df = df[(df['Volume'] < upper_bound) & (df['Volume'] > lower_bound)]
    df=df[(np.abs(df['Volume']) < (3 * df['Volume'].std()))]
    df = df.dropna()
    df=df.drop_duplicates()
    x = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    return (df,x,y)

# df,x,y= Cleaning('IDA STOCK 1.xlsx')

# df = pd.read_excel('IDA STOCK 1.xlsx')
def Split(file_path):
    df, x, y = Cleaning(file_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=False)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=False)

    model = LinearRegression()
    model.fit(x_train, y_train)

    print(model.score(x_test, y_test) * 100)

    # Make predictions on the testing set
    y_pred = model.predict(x)

    # Evaluate the model performance
    mse = mean_squared_error(y, y_pred)
    print('Mean squared error:', mse)

    # Convert 'Date' to datetime
    dates = pd.to_datetime(df['Date'])
    dates_train, dates_test = train_test_split(dates, test_size=0.25, shuffle=False)

    # Plotting the results
    plt.plot(dates, y, label='Actual Close Price')
    plt.plot(dates, y_pred, color="pink", label='Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Price Over Time')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()


def Trainer(file_path):
    model = LinearRegression()
    df, x, y = Cleaning(file_path)
    model.fit(x, y)
    return model

def Tester(file_path , model):
    df,x,y=Cleaning(file_path)

    # Make predictions on the testing set
    y_pred = model.predict(x)
    print(f"Model Score is {model.score(x, y) * 100}")

    # Evaluate the model performance
    mse = mean_squared_error(y, y_pred)
    print(f'Mean squared error:{mse} \n')

    # Convert 'Date' to datetime
    dates = pd.to_datetime(df['Date'])
    # dates_train, dates_test = train_test_split(dates, test_size=0.25, shuffle=False)
    ax = plt.axes()
    ax.set_facecolor("black")

    # Plotting the results
    plt.plot(dates, y,color="cyan" ,label='Actual Close Price')
    plt.plot(dates, y_pred, color="yellow", label='Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Price Over Time')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()

# Split("IDA STOCK 1.xlsx")
model=Trainer("IDA STOCK 1.xlsx")
Tester("IDA STOCK 1.xlsx",model)
Tester("stock2.xlsx",model)
Tester("stock3.xlsx",model)