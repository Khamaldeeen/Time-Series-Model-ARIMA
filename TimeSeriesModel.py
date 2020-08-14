import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from random import choice
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf
sns.set_style('darkgrid')


class TimeSeriesModel:
    #Specifies the store number, item number and any consecutive 3 months of interest in the same year.
    def __init__(self, data, f_mnth, s_mnth, t_mnth, store_no, item_no):
        df = pd.read_csv(data)
        df = df[df['date'].str.contains(f_mnth) | df['date'].str.contains(s_mnth) | df['date'].str.contains(t_mnth)].reset_index(drop=True)
        df = df.set_index('date')
        df = df[df['store'] == store_no]
        df = df[df['item'] == item_no]
        self.df = df
        self.st = store_no
        self.it = item_no
    
    #Returns the dataframe specified.
    def dataframe(self):
        return self.df
    
    #Plots the graph of the containing dataset.
    def lineplot(self, win_no ):
        df = self.df
        df_sales = df['sales']
        lis = [df_sales]
        for item in win_no:
            col_name = 'MA for %s days' %(str(item))
            df[col_name] = df['sales'].rolling(window=item).mean()
            lis.append(df[col_name])
        scale = len(lis)/2
        plt.figure(figsize=(20, 10*scale))
        x = 1
        for i in lis:
            col = '#' + ''.join([choice('0123456789ABCEDF') for j in range(6)])
            plt.subplot((len(win_no) + 1), 1, x)
            i.plot(legend=True, color=col)
            if x == 1:
                plt.title(f'Sales Trend without Moving average of store {self.st} and item {self.it}.')
            else:
                plt.title(f'Sales Trend with Moving Average of store {self.st} and item {self.it}.')
            x += 1

    #Plots the Kernel Density and Bargraph distribution of the percent change in the daily return.
    def distplot(self):
        df = self.df
        df['Daily Return'] = df['sales'].pct_change()
        sns.distplot(df['Daily Return'].dropna(), bins=100, color='Green')

    #Returns the dataframe of the percentage change of daily returns.
    def changedf(self):
        df = self.df
        df['Daily Return'] = df['sales'].pct_change()
        return df

    #The auto correlation plot is determined from the chart and our P values is chosen.
    def autocorr(self):
        df = self.df['sales']
        series = pd.Series(df)
        pd.plotting.autocorrelation_plot(series)
        plt.show()

    #A partial autocorreation plot is shown to determine an appropriate Moving Average value.
    def partialcorr(self, no_lag):
        df = self.df
        ser_pac = pd.Series(df['sales'])
        plot_pacf(ser_pac, lags = no_lag)

    #The model is fitted with this function
    def arima(self, order):
        df = self.df
        model = ARIMA(df['sales'], order)
        model_fit = model.fit(disp=0)
        f_mod = model_fit.summary()
        print(f_mod)

    #The residual plot is shown here while also specifying model order.
    def resid(self, order):
        df = self.df
        model = ARIMA(df['sales'], order)
        model_fit = model.fit(disp=0)
        res = pd.DataFrame(model_fit.resid)
        res.plot()
        print(res.plot(kind='kde'))
        print(res.describe())
    
    #Plot of the predicted or forcasted sales along side the actual sales data. 
    def forcast(self, train_size, order):
        df = self.df
        df = df['sales']
        X = df.values 
        size = int(len(X) * train_size )
        train, test = X[0 : size], X[size : len(X)]
        hist = [x for x in train]
        pred = []
        for i in range(len(test)):
            m = ARIMA(hist, order)
            m_fit = m.fit(disp=0)
            output = m_fit.forecast()
            yhat = output[0]
            pred.append(yhat)
            orig = test[i]
            hist.append(orig)
            #print('Predicted = %f, Actual Value = %f' % (yhat, orig))
        err = mean_squared_error(test, pred)
        print(f"MSE of Test data = {err}" )
        plt.plot(test)
        plt.plot(pred, color = 'Red')
        return pred, test

