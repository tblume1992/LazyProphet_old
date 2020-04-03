# LazyProphet
Time Series decomp via gradient boosting with a couple different estimators of trend:
  * ridge: approximates trend via a global fit from a ridge regression (don't really need ridge since we are boosting but oh well)
  * linear: approximates trend via a local linear changepoint model done using binary segmented regressions to minimize MAE
  * mean: approximates trend via local mean change point model
  
Seasonality is initially a naive average over every freq number of points.  After boosting is complete we take the top 15 (or whatever you select as seasonal_smoothing) components from fft. Set to 0 or None for no seasonality.
 
Notes:
1.  Number of gradient boosting rounds can be set to a max but once our cost function is minimized it will stop
2.  You probably want to always have ols_constant = False for linear estimator
3.  We can approximate where splits should occur for our local estimators (mean and linear) which speeds things up quite a bit 
4.  The regularization parameter effects the number of boosting rounds whereas l2 just effects the ridge regression regularization



Quick example: 

```python
import quandl
import fbprophet
import pandas as pd
import matplotlib.pyplot as plt
import LazyProphet as lp

#Get bitcoin data
data = quandl.get("BITSTAMP/USD")
y = data['Low']
y = y[-730:]
df = pd.DataFrame(y)
df['ds'] = y.index
#adjust to make ready for Prophet
df.columns = ['y', 'ds']
model = fbprophet.Prophet()
model.fit(df)
forecast = model.predict(df)

#create Lazy Prophet class
boosted_model = lp.LazyProphet(freq = 365, 
                            estimator = 'linear', 
                            max_boosting_rounds = 50,
                            approximate_splits = True,
                            regularization = 1.2)
#Fits on just the time series
#returns a dictionary with the decomposition
output = boosted_model.fit(y)
#plot forecasts vs actual
tsboosted_ = output['yhat']
proph = forecast['yhat']
plt.plot(tsboosted_, label = 'Lazy', color = 'black')
proph.index = tsboosted_.index
plt.plot(y, label = 'Actual')
plt.plot(proph, label = 'Prophet')
plt.legend()
plt.show()
```
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/lazy_image_1.png?raw=true "Output 1")

An example using ridge and looking at the trend and seasonality decomp:
```python
import quandl
import fbprophet
import pandas as pd
import matplotlib.pyplot as plt
import LazyProphet as lp

#Get bitcoin data
data = quandl.get("BITSTAMP/USD")
y = data['Low']
y = y[-730:]
df = pd.DataFrame(y)
df['ds'] = y.index
#adjust to make ready for Prophet
df.columns = ['y', 'ds']
model = fbprophet.Prophet()
model.fit(df)
forecast = model.predict(df)

#create Lazy Prophet class
boosted_model = lp.LazyProphet(freq = 365, 
                            estimator = 'ridge', 
                            max_boosting_rounds = 50,
                            approximate_splits = True,
                            regularization = 1.2)
#Fits on just the time series
#returns a dictionary with the decomposition
output = boosted_model.fit(y)
#plot forecasts vs actual
tsboosted_ = output['yhat']
proph = forecast['yhat']
plt.plot(tsboosted_, label = 'Lazy', color = 'black')
proph.index = tsboosted_.index
plt.plot(y, label = 'Actual')
plt.plot(proph, label = 'Prophet')
plt.legend()
plt.show()
#plot trend
plt.plot(forecast['trend'], label = 'Prophet')
plt.plot(output['trend'].reset_index(drop = True), label = 'Lazy')
plt.plot(y.reset_index(drop = True))
plt.legend()
plt.show()
#plot seasonality
plt.plot(forecast['additive_terms'], label = 'Prophet')
plt.plot(output['seasonality'].reset_index(drop = True), label = 'Lazy')
plt.legend()
plt.show()
```
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/lazy_ridge_1.png?raw=true "Output 1")
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/lazy_ridge_trend.png?raw=true "Output 1")
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/lazy_ridge_seasonality.png?raw=true "Output 1")

An example using mean changepoints:
```python
import quandl
import fbprophet
import pandas as pd
import matplotlib.pyplot as plt
import LazyProphet as lp

#Get bitcoin data
data = quandl.get("BITSTAMP/USD")
y = data['Low']
y = y[-730:]
df = pd.DataFrame(y)
df['ds'] = y.index
#adjust to make ready for Prophet
df.columns = ['y', 'ds']
model = fbprophet.Prophet()
model.fit(df)
forecast = model.predict(df)

#create Lazy Prophet class
boosted_model = lp.LazyProphet(freq = 365, 
                            estimator = 'mean', 
                            max_boosting_rounds = 50,
                            approximate_splits = True,
                            regularization = 1.2)
#Fits on just the time series
#returns a dictionary with the decomposition
output = boosted_model.fit(y)
#plot forecasts vs actual
tsboosted_ = output['yhat']
proph = forecast['yhat']
plt.plot(tsboosted_, label = 'Lazy', color = 'black')
proph.index = tsboosted_.index
plt.plot(y, label = 'Actual')
plt.plot(proph, label = 'Prophet')
plt.legend()
plt.show()
#plot trend
plt.plot(forecast['trend'], label = 'Prophet')
plt.plot(output['trend'].reset_index(drop = True), label = 'Lazy')
plt.plot(y.reset_index(drop = True))
plt.legend()
plt.show()
```
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/lazy_mean_1.png?raw=true "Output 1")
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/lazy_mean_trend.png?raw=true "Output 1")

What is the impact of the coronavirus?
```python
import quandl
import fbprophet
import pandas as pd
import LazyProphet as lp

#Get bitcoin data
data = quandl.get("BITSTAMP/USD")
y = data['Low']
y = y[-730:]

#create Lazy Prophet class
boosted_model = lp.LazyProphet(freq = None, 
                            estimator = 'mean', 
                            approximate_splits = True)
#Fits on just the time series
#returns a dictionary with the decomposition
output = boosted_model.fit(y)

#Potential impact of coronavirus with a 'still normal' date of Feb 1st
pct_change = output['trend'].loc[(output['trend'].index > '2020-02-01')].pct_change()
pct_change = pct_change.replace(to_replace=0, method='ffill')
impact = np.mean(pct_change)
print(f'Maybe like {int(impact*100)} percent?')
```
