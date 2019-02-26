
# coding: utf-8

# Problem Statement:
# Pick up the following stocks and generate forecasts accordingly Stocks:
# 
# 1.NASDAQ.AAPL
# 
# 2.NASDAQ.ADP
# 
# 3.NASDAQ.CBOE
# 
# 4.NASDAQ.CSCO
# 
# 5.NASDAQ.EBAY

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA, ARMAResults
import datetime
import sys
import seaborn as sns
import statsmodels
import statsmodels.stats.diagnostic as diag
from statsmodels.tsa.stattools import adfuller
from scipy.stats.mstats import normaltest
from matplotlib.pyplot import acorr
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Load data
df = pd.read_csv('C:/Users/HP/data_stocks.csv')
df.head()


# In[3]:


# display five records in the dataframe
df.head().transpose()


# In[4]:


# make a list of columns
stock_features =['NASDAQ.AAPL','NASDAQ.ADP','NASDAQ.CBOE','NASDAQ.CSCO','NASDAQ.EBAY']
col_list = ['DATE'] + stock_features
df1 = df[col_list]
df1.head()


# In[5]:


df1.info()


# In[6]:


#Checking for null values if any
df1.isnull().sum()


# In[7]:


df1 =df1.copy()
df1['DATE'] =  pd.to_datetime(df1['DATE'])
df1.tail()


# In[8]:


df1.head()


# In[9]:


df1 = df1.copy()
df1['Month'] = df1['DATE'].dt.date


# In[10]:


df1.head()


# In[11]:


col_list = ['Month']+ stock_features
df2 = df1[col_list]
df2.head()


# In[12]:


df2.isnull().sum()


# In[13]:


df2.describe().transpose()


# In[14]:


final = df2.copy()
final['Month']=pd.to_datetime(final['Month'])


# In[15]:


#Time Series Forecasting for NASDAQ.AAPL

df_AAPL = final[['Month',stock_features[0]]]
df_AAPL.head()


# In[16]:


df_AAPL.set_index('Month',inplace=True)
df_AAPL.head()


# In[17]:


df_AAPL.index


# In[18]:


#Summary Statistics

df_AAPL.describe().transpose()


# In[19]:


#Step 2 : Visualize the Data

import seaborn as sns
sns.set_style('whitegrid')
df_AAPL.plot()
plt.title('Time Series Plot for NASDAQ_AAPL')
plt.show()


# In[20]:


#Plotting Rolling Statistics and check for stationarity :
#The function will plot the moving mean or moving Standard Deviation. This is still visual method

#NOTE: Moving mean and Moving standard deviation —  At any instant ‘t’, 
#we take the mean/std of the last year which in this case is 12 months)


# In[21]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(timeseries)
    print('\nAugmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    for k,v in result[4].items():
        print('Crtical {} : value {}'.format(k,v))
    
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
test_stationarity(df_AAPL['NASDAQ.AAPL'])


# In[22]:


#Note:This is not stationary because :Mean is increasing even though the Std is small.

#Test stat is > critical value.
#Note: the signed values are compared and the absolute values.


# In[23]:


#MAKE THE TIME SERIES STATIONARY
#There are two factors that make a time series non-stationary. They are:

#Trend: non-constant mean
#Seasonality: Variation at specific time-frames




#Differencing
#The first difference of a time series is the series of changes from one period to the next. 
#We can do this easily with pandas. 
#You can continue to take the second difference, third difference, and so on until your data is stationary.


# In[24]:


#First Difference

df_AAPL = df_AAPL.copy()
df_AAPL.loc[:,'First_Difference'] = df_AAPL['NASDAQ.AAPL'] - df_AAPL['NASDAQ.AAPL'].shift(1)
df_AAPL.head()


# In[25]:


df_AAPL = df_AAPL.copy()
df_AAPL.dropna(inplace=True)


# In[26]:


#Test Staionarity

test_stationarity(df_AAPL['First_Difference'])


# In[27]:


#Seasonal Decomposition

from statsmodels.tsa.seasonal import seasonal_decompose
plt.figure(figsize=(11,8))
decomposition = seasonal_decompose(df_AAPL['NASDAQ.AAPL'],freq=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(df_AAPL['NASDAQ.AAPL'],label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')


# In[28]:


#Note:
#The data is seasonal as interpreted from the Seasonal plot of seasonal decomposition.


# In[29]:


ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)


# In[30]:


#Note :This is stationary because:

#Test statistic is lower than critical values.

#The Mean and Std Variations have small variations with time.


# In[31]:


#Autocorrelation and Partial Autocorrelation Plots

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plt.figure(figsize=(20,8))
fig_first = plot_acf(df_AAPL["First_Difference"],lags=30,title='Autocorrelation-NASDAQ.AAPL')


# In[32]:


plt.figure(figsize=(20,8))
fig_pacf_first = plot_pacf(df_AAPL["First_Difference"],lags=30,title='Partial Autocorrelation-NASDAQ.AAPL')


# In[33]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df_AAPL['First_Difference'])


# In[34]:


#Forecasting a Time Series
#Auto Regressive Integrated Moving Average(ARIMA) —

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[35]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_AAPL['First_Difference'].iloc[30:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_AAPL['First_Difference'].iloc[30:], lags=40, ax=ax2)


# In[36]:


lag_acf = acf(df_AAPL['First_Difference'],nlags=80)
lag_pacf = pacf(df_AAPL['First_Difference'],nlags=80,method='ols')


# In[37]:


plt.figure(figsize=(10,10))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_AAPL['First_Difference'])),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_AAPL['First_Difference'])),linestyle='--',color='gray')

plt.title('Autocorrelation')

plt.subplot(122)

plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_AAPL['First_Difference'])),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_AAPL['First_Difference'])),linestyle='--',color='gray')

plt.title('Partial Autocorrelation')

plt.tight_layout()


# In[38]:


#Note- The two dotted lines on either sides of 0 are the confidence intervals.

#These can be used to determine the ‘p’ and ‘q’ values as:

#p: The first time where the PACF crosses the upper confidence interval, here its close to 0. hence p = 0.

#q: The first time where the ACF crosses the upper confidence interval, here its close to 0. hence p = 0.


# In[39]:


#Using the Seasonal ARIMA model

model= sm.tsa.statespace.SARIMAX(df_AAPL['NASDAQ.AAPL'],order=(0,1,0),seasonal_order=(0,1,0,12))
results = model.fit()
print(results.summary())


# In[40]:


results.resid.plot()


# In[41]:


results.resid.plot(kind='kde')


# In[42]:


df_AAPL = df_AAPL.copy()
df_AAPL['Forecast'] = results.predict()

df_AAPL.head()


# In[43]:


#Prediction of Future Values

df_AAPL[['NASDAQ.AAPL','Forecast']].plot(figsize=(12,8))


# In[44]:


results.forecast(steps=10)


# In[45]:


results.predict(start=41264,end=41274)


# In[46]:


results.predict(start=41264,end=41274)


# In[47]:


#Accuracy of the Forecast using MSE-Mean Squared Error

from sklearn.metrics import mean_squared_error,mean_absolute_error
print('Mean Squared Error NASDAQ.AAPL -', mean_squared_error(df_AAPL['NASDAQ.AAPL'],df_AAPL['Forecast']))
print('Mean Absolute Error NASDAQ.AAPL -', mean_absolute_error(df_AAPL['NASDAQ.AAPL'],df_AAPL['Forecast']))


# In[48]:


#Time Series Forecasting for NASDAQ.ADP

df_ADP = final[['Month',stock_features[1]]]
df_ADP.head()


# In[49]:


df_ADP.set_index('Month',inplace=True)
df_ADP.head()


# In[50]:


#Visualize Data

df_ADP.plot()
plt.title('Time Series Plot for NASDAQ_ADP')
plt.show()


# In[51]:


test_stationarity(df_ADP['NASDAQ.ADP'])


# In[52]:


#MAKING THE TIME SERIES STATIONARY

#Differencing

df_ADP = df_ADP.copy()
df_ADP['First_Difference'] = df_ADP['NASDAQ.ADP'] - df_ADP['NASDAQ.ADP'].shift(1)
df_ADP.head()


# In[53]:


df_ADP.dropna(inplace=True)
test_stationarity(df_ADP['First_Difference'])
#Now subtract the rolling mean from the original series


# In[54]:


#Seasonal Decomposition

from statsmodels.tsa.seasonal import seasonal_decompose
plt.figure(figsize=(11,8))
decomposition = seasonal_decompose(df_ADP['First_Difference'],freq=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(df_ADP['First_Difference'],label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')


# In[55]:


#Note: The data for NASDAQ.ADP is seasonal as interpreted from the seasonal plot of seasonal decomposition.


# In[56]:


ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)


# In[57]:


#Note : This is stationary because:

#Test statistic is lower than 1% critical values.

#The mean and std variations have small variations with time


# In[58]:


#Autocorrelation and Partial Corelation plot

fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_ADP['First_Difference'].iloc[38:], lags=80, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_ADP['First_Difference'].iloc[38:], lags=80, ax=ax2)


# In[59]:


lag_acf = acf(df_ADP['First_Difference'],nlags=80)
lag_pacf = pacf(df_ADP['First_Difference'],nlags=80,method='ols')


# In[60]:


plt.figure(figsize=(20,8))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_ADP['First_Difference'])),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_ADP['First_Difference'])),linestyle='--',color='gray')

plt.title('Autocorrelation')

plt.subplot(122)

plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_ADP['First_Difference'])),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_ADP['First_Difference'])),linestyle='--',color='gray')

plt.title('Partial Autocorrelation')


# In[61]:


#Note- The two dotted lines on either sides of 0 are the confidence intervals.

#These can be used to determine the ‘p’ and ‘q’ values as:

#p: The first time where the PACF crosses the upper confidence interval, here its close to 0. hence p = 0.

#q: The first time where the ACF crosses the upper confidence interval, here its close to 0. hence p = 0.


# In[62]:


model= sm.tsa.statespace.SARIMAX(df_ADP['NASDAQ.ADP'],order=(0,1,0),seasonal_order=(0,1,0,12))
results = model.fit()
print(results.summary())


# In[63]:


plt.plot(results.resid)


# In[64]:


sns.set_style('whitegrid')
sns.kdeplot(results.resid)


# In[65]:


df_ADP['Forecast'] = results.predict()
df_ADP[['NASDAQ.ADP','Forecast']].tail()


# In[66]:


results.forecast(steps=10)


# In[67]:


results.predict(start=41264,end=41275)


# In[68]:


df_ADP[['NASDAQ.ADP','Forecast']].plot(figsize=(20,8))


# In[69]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
print('Mean Squared Error NASDAQ.AAPL -', mean_squared_error(df_ADP['NASDAQ.ADP'],df_ADP['Forecast']))
print('Mean Absolute Error NASDAQ.AAPL -', mean_absolute_error(df_ADP['NASDAQ.ADP'],df_ADP['Forecast']))


# In[70]:


#Times Series Forecasting for 'NASDAQ.CBOE'

df_CBOE= final[['Month',stock_features[2]]]
print(df_CBOE.head())
df_CBOE.set_index('Month',inplace=True)
print(df_CBOE.head())

df_CBOE.plot()
plt.title('Time Series Plot for NASDAQ_CBOE')
plt.show()
#test Stationarity
test_stationarity(df_CBOE['NASDAQ.CBOE'])


# In[71]:


#MAKING THE TIME SERIES STATIONARY

#Differencing

df_CBOE = df_CBOE.copy()
df_CBOE.head()


# In[72]:


df_CBOE['First_Difference'] = df_CBOE['NASDAQ.CBOE'] - df_CBOE['NASDAQ.CBOE'].shift(1)
df_CBOE.head()


# In[73]:


df_CBOE.dropna(inplace=True)


# In[74]:


#Test Seasonality

test_stationarity(df_CBOE['First_Difference'])


# In[75]:


#Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
plt.figure(figsize=(11,8))
decomposition = seasonal_decompose(df_CBOE['NASDAQ.CBOE'],freq=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(df_CBOE['NASDAQ.CBOE'],label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')


# In[76]:


ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)


# In[77]:


#Autocorrelation and Partial Corelation plot

fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_CBOE['First_Difference'].iloc[26:], lags=80, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_CBOE['First_Difference'].iloc[26:], lags=80, ax=ax2)


# In[78]:


lag_acf = acf(df_CBOE['First_Difference'],nlags=80)
lag_pacf = pacf(df_CBOE['First_Difference'],nlags=80,method='ols')

plt.figure(figsize=(11,8))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_CBOE['First_Difference'])),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_CBOE['First_Difference'])),linestyle='--',color='gray')

plt.title('Autocorrelation')

plt.subplot(122)

plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_CBOE['First_Difference'])),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_CBOE['First_Difference'])),linestyle='--',color='gray')

plt.title('Partial Autocorrelation')

plt.tight_layout()


# In[79]:


#Note- The two dotted lines on either sides of 0 are the confidence intervals.

#These can be used to determine the ‘p’ and ‘q’ values as:

#p: The first time where the PACF crosses the upper confidence interval, here its close to 0. hence p = 0.

#q: The first time where the ACF crosses the upper confidence interval, here its close to 0. hence p = 0.


# In[80]:


# fit model
model= sm.tsa.statespace.SARIMAX(df_CBOE['NASDAQ.CBOE'],order=(0,1,0),seasonal_order=(0,1,0,12))
results = model.fit()
print(results.summary())
print(results.forecast())
df_CBOE['Forecast'] = results.predict()
df_CBOE[['NASDAQ.CBOE','Forecast']].plot(figsize=(20,8))
plt.show()


# In[81]:


results.forecast(steps=10)


# In[82]:


results.predict(start=41264,end=41273)


# In[83]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
print('Mean Squared Error NASDAQ.CBOE -', mean_squared_error(df_CBOE['NASDAQ.CBOE'],df_CBOE['Forecast']))
print('Mean Absolute Error NASDAQ.CBOE -', mean_absolute_error(df_CBOE['NASDAQ.CBOE'],df_CBOE['Forecast']))


# In[84]:


#Time Series ForeCasting for 'NASDAQ.CSCO'

df_CSCO = final[['Month',stock_features[3]]]
print(df_CSCO.head())
df_CSCO.set_index('Month',inplace=True)
print(df_CSCO.head())
df_CSCO.plot()
plt.title("Time Series Plot for NASDAQ.CSCO")
plt.show()
#Test Staionarity
test_stationarity(df_CSCO['NASDAQ.CSCO'])


# In[85]:


#MAKING TIME SERIES STATIONARY

#Differencing

df_CSCO = df_CSCO.copy()
df_CSCO['First_Difference'] = df_CSCO['NASDAQ.CSCO'] - df_CSCO['NASDAQ.CSCO'].shift(1)
df_CSCO.dropna(inplace=True)
test_stationarity(df_CSCO['First_Difference'])


# In[86]:


#Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
plt.figure(figsize=(11,8))
decomposition = seasonal_decompose(df_CSCO['NASDAQ.CSCO'],freq=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(df_CSCO['NASDAQ.CSCO'],label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')


# In[87]:


ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)


# In[88]:


#Note : This is stationary because:

#Test statistic is lower than critical values.

#The mean and std variations have small variations with time


# In[89]:


#Auto Corealtion and Partial Autocorelation Plots

fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_CSCO['First_Difference'].iloc[46:], lags=80, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_CSCO['First_Difference'].iloc[46:], lags=80, ax=ax2)


# In[90]:


lag_acf = acf(df_CSCO['First_Difference'],nlags=80)
lag_pacf = pacf(df_CSCO['First_Difference'],nlags=80,method='ols')


# In[91]:


plt.figure(figsize=(20,8))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_CSCO['First_Difference'])),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_CSCO['First_Difference'])),linestyle='--',color='gray')

plt.title('Autocorrelation')

plt.subplot(122)

plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_CSCO['First_Difference'])),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_CSCO['First_Difference'])),linestyle='--',color='gray')

plt.title('Partial Autocorrelation')

plt.tight_layout()


# In[92]:


#Note- The two dotted lines on either sides of 0 are the confidence intervals.

#These can be used to determine the ‘p’ and ‘q’ values as:

#p: The first time where the PACF crosses the upper confidence interval, here its close to 0. hence p = 0.

#q: The first time where the ACF crosses the upper confidence interval, here its close to 0. hence p = 0.


# In[93]:


# fit model
model= sm.tsa.statespace.SARIMAX(df_CSCO['NASDAQ.CSCO'],order=(0,1,0),seasonal_order=(0,1,0,12))
results = model.fit()
print(results.summary())
df_CSCO['Forecast'] = results.predict()
df_CSCO[['NASDAQ.CSCO','Forecast']].plot(figsize=(20,8))
plt.show()


# In[94]:


df_CSCO.head()


# In[95]:


results.forecast(steps=10)


# In[96]:


results.predict(start=41264,end=41275)


# In[97]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
print('Mean Squared Error NASDAQ.CSCO -', mean_squared_error(df_CSCO['NASDAQ.CSCO'],df_CSCO['Forecast']))
print('Mean Absolute Error NASDAQ.CSCO -', mean_absolute_error(df_CSCO['NASDAQ.CSCO'],df_CSCO['Forecast']))


# In[98]:


#Time Series Forecasting for NASDAQ.EBAY

df_EBAY = final[['Month',stock_features[4]]]
print(df_EBAY.head())
df_EBAY.set_index('Month',inplace=True)
print(df_EBAY.head())
df_EBAY.plot()
plt.title("Time Series Plot for NASDAQ.EBAY")
plt.show()
#Test Staionarity
test_stationarity(df_EBAY['NASDAQ.EBAY'])


# In[99]:


#MAKING TIME SERIES STATIONARY
#Differencing

df_EBAY = df_EBAY.copy()
df_EBAY['First_Difference'] = df_EBAY['NASDAQ.EBAY'] - df_EBAY['NASDAQ.EBAY'].shift(1)
df_EBAY.dropna(inplace=True)
#test Stationarity
test_stationarity(df_EBAY['NASDAQ.EBAY'])


# In[100]:


#Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
plt.figure(figsize=(11,8))
decomposition = seasonal_decompose(df_EBAY['NASDAQ.EBAY'],freq=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(df_EBAY['NASDAQ.EBAY'],label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')


# In[101]:


ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)


# In[102]:


# Note : This is stationary because:

#Test statistic is lower than critical values.

#The mean and std variations have small variations with time


# In[103]:


#Autocorealtion plot and Partial Autocorelation plots

fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_EBAY['First_Difference'].iloc[47:], lags=80, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_EBAY['First_Difference'].iloc[47:], lags=80, ax=ax2)


# In[104]:


lag_acf = acf(df_EBAY['First_Difference'],nlags=80)
lag_pacf = pacf(df_EBAY['First_Difference'],nlags=80,method='ols')


# In[105]:


plt.figure(figsize=(20,8))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_EBAY['First_Difference'])),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_EBAY['First_Difference'])),linestyle='--',color='gray')

plt.title('Autocorrelation')

plt.subplot(122)

plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_EBAY['First_Difference'])),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_EBAY['First_Difference'])),linestyle='--',color='gray')

plt.title('Partial Autocorrelation')

plt.tight_layout()


# In[106]:


#Note- The two dotted lines on either sides of 0 are the confidence intervals.

#These can be used to determine the ‘p’ and ‘q’ values as:

#p: The first time where the PACF crosses the upper confidence interval, here its close to 0. hence p = 0.

#q: The first time where the ACF crosses the upper confidence interval, here its close to 0. hence p = 0.


# In[107]:


# fit model
model= sm.tsa.statespace.SARIMAX(df_EBAY['NASDAQ.EBAY'],order=(0,1,0),seasonal_order=(0,1,0,12))
results = model.fit()
print(results.summary())
df_EBAY['Forecast'] = results.predict()
df_EBAY[['NASDAQ.EBAY','Forecast']].plot(figsize=(20,8))
plt.show()


# In[108]:


df_EBAY.head()


# In[109]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
print('Mean Squared Error NASDAQ.EBAY -', mean_squared_error(df_EBAY['NASDAQ.EBAY'],df_EBAY['Forecast']))
print('Mean Absolute Error NASDAQ.EBAY -', mean_absolute_error(df_EBAY['NASDAQ.EBAY'],df_EBAY['Forecast']))


# In[110]:


results.forecast(steps=10)


# In[111]:


results.predict(start=41265,end=41275)


# CONCLUSION :
# The predicted stock prices values have been stored in the Forecast Columns of the each stock entity dataframe
