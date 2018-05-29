import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


HangSengIndex = pd.read_csv("^HSI.csv")
DowJonesIndex = pd.read_csv("^DJI.csv")
NasdaqIndex = pd.read_csv("^IXIC.csv")
SandP500Index = pd.read_csv("^GSPC.csv")

DowJonesIndex.columns = ['Date', 'DJI Open', 'DJI High', 'DJI Low', 'DJI Close', 'DJI Adj Close', 'DJI Volume']
NasdaqIndex.columns = ['Date', 'IXIC Open', 'IXIC High', 'IXIC Low', 'IXIC Close', 'IXIC Adj Close', 'IXIC Volume']
SandP500Index.columns = ['Date', 'GSPC Open', 'GSPC High', 'GSPC Low', 'GSPC Close', 'GSPC Adj Close', 'GSPC Volume']
HangSengIndex.columns = ['Date', 'HSI Open', 'HSI High', 'HSI Low', 'HSI Close', 'HSI Adj Close', 'HSI Volume']

#drop HSI where Open==High==Low==Close
HangSengIndex = HangSengIndex.iloc[203:]

df = pd.merge(DowJonesIndex,NasdaqIndex,on='Date', how='inner')
df = pd.merge(df,SandP500Index,on='Date', how='inner')
df = pd.merge(df,HangSengIndex,on='Date', how='inner')
df= df.sort_values(by='Date')


df['HSI Low'] = df['HSI Low'].replace(to_replace='null', value='0') #this is per column operation
#df = df[df['HSI High'] != 'null']
df = df.dropna()
#df = df.reset_index(drop=True)
df = df.iloc[:,1:]


df = df.astype('int64')


#extract from 1 to t as training/test dataset, then remove row t
open_tplus1 = df['HSI Open'][1:]
high_tplus1 = df['HSI High'][1:]
low_tplus1 = df['HSI Low'][1:]
close_tplus1 = df['HSI Close'][1:]

df = df.iloc[:-1] #remove last row t



X = df.iloc[:,[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22]]
y = open_tplus1
#df.loc[:,['HSI Open','HSI High','HSI Low','HSI Close']]

#X = X.values
#y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lm = linear_model.LinearRegression()
model_open = lm.fit(X_train ,y_train)
y_open_pred = model_open.predict(X_test)
#mse_regression = mse(y_test, y_open_pred)
mae_open_regression = mae(y_test, y_open_pred)
#print(mse_regression)
print(mae_open_regression)

import plotly.plotly as py
import plotly.graph_objs as go

#upper_bound = go.Scatter(
#   name='Upper Bound',
#    x=X_test.index,
#    y=y_high_pred
#    mode='lines',
#    marker=dict(color="444"),
#    line=dict(width=0),
#    fillcolor='rgba(68, 68, 68, 0.3)',
#    fill='tonexty' )

y = high_tplus1
#df.loc[:,['HSI Open','HSI High','HSI Low','HSI Close']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lm = linear_model.LinearRegression()
model = lm.fit(X_train ,y_train)

y_high_pred = lm.predict(X_test)

#mse_regression = mse(y_test, y_high_pred)
mae_high_regression = mae(y_test, y_high_pred)

#print(mse_regression)
print(mae_high_regression)

y = low_tplus1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lm = linear_model.LinearRegression()
model = lm.fit(X_train ,y_train)

y_low_pred = lm.predict(X_test)

#mse_regression = mse(y_test, y_low_pred)
mae_low_regression = mae(y_test, y_low_pred)

#print(mse_regression)
print(mae_low_regression)

y = close_tplus1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lm = linear_model.LinearRegression()
model_open = lm.fit(X_train ,y_train)

y_close_pred = lm.predict(X_test)

#mse_regression = mse(y_test, y_close_pred)
mae_close_regression = mae(y_test, y_close_pred)

#print(mse_regression)
print(mae_close_regression)

# Initialize the `signals` DataFrame with the `signal` column

#backtest = pd.read_csv('backtest - 1.csv')
X_backtest= X.iloc[:-300]
signals = pd.DataFrame(index=X_backtest.index)
signals['long signal'] = 0.0

y_backtest_open = open_tplus1.iloc[:,22]
y_backtest_high = high_tplus1.iloc[:,23]
y_backtest_low = low_tplus1.iloc[:,24]
y_backtest_close = close_tplus1.iloc[:,25]

y_backtest_open_pred = model_open.predict(X_backtest)

signals['long signal'] = y_close_pred  > y_open_pred + mae_open_regression
import numpy as np
signals = signals.reset_index()
#np.where(signals['long signal'])[0]
#backtestx = X_test.reset_index()
backtesty1 = open_tplus1.reset_index()
backtesty2 = high_tplus1.reset_index()
backtesty3 = low_tplus1.reset_index()
backtesty4 = close_tplus1.reset_index()
#dfgg = pd.merge(signals,backtestx, on='index', how='inner')
dflong = pd.merge(signals,backtesty1,on='index', how='inner')
dflong = pd.merge(dflong,backtesty4,on='index', how='inner')

dflong.groupby('long signal').sum()

signals['short signal'] = y_open_pred>y_low_pred + mae_low_regression
#np.where(signals['short signal'])[0]
dfshort = pd.merge(signals, backtesty1, on='index', how='inner')
dfshort = pd.merge(dfshort, backtesty3)

dfshort.groupby('short signal').sum()

