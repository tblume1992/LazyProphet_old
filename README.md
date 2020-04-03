# LazyProphet
Quick example: 

```python
import quandl
import fbprophet
import pandas as pd
import matplotlib.pyplot as plt

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
boosted_model = LazyProphet(freq = 365, 
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

An example using mean changepoints:
```python
import quandl
import fbprophet
import pandas as pd
import matplotlib.pyplot as plt

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
boosted_model = LazyProphet(freq = 365, 
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
