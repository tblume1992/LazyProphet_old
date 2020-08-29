# LazyProphet

```
pip install LazyProphet
```
Time Series decomp via gradient boosting with a couple different estimators of trend:
  * ridge: approximates trend via a global fit from a polynomial ridge regression (don't really need ridge since we are boosting but oh well)
  * linear: approximates trend via a local linear changepoint model done using binary segmented regressions to minimize MAE
  * mean: approximates trend via local mean change point model
  
Seasonality can be naive averaging over freq number of time periods or 'harmonic' which calculates seasonality similarly to Prophet using fourier series.
 
Notes:
1.  Number of gradient boosting rounds can be set to a max but once our cost function is minimized it will stop unless a minimum is set
2.  You probably want to always have ols_constant = False for linear estimator
3.  We can approximate where splits should occur for our local estimators (mean and linear) which speeds things up quite a bit 
4.  The regularization parameter effects the number of boosting rounds whereas l2 just effects the ridge regression regularization

Basic flow of the algorithm:



![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lp_flow.PNG?raw=true "Output 1")

# Some basic examples: 

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
                               approximate_splits = True,
                               regularization = 1.2,
                               global_cost = 'maicc',
                               split_cost = 'mse',
                               seasonal_regularization = 'auto',
                               trend_dampening = 0,
                               max_boosting_rounds = 50,
                               exogenous = None
                                    )
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
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_1.png?raw=true "Output 1")
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_linear_trend.png?raw=true "Output 1")
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_linear_seasonality.png?raw=true "Output 1")

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
                               approximate_splits = True,
                               regularization = 1.2,
                               global_cost = 'maicc',
                               split_cost = 'mse',
                               seasonal_regularization = 'auto',
                               trend_dampening = 0,
                               max_boosting_rounds = 50,
                               exogenous = None)
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
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_ridge.png?raw=true "Output 1")
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_ridge_trend.png?raw=true "Output 1")
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_ridge_seasonality.png?raw=true "Output 1")

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
#plot seasonality
plt.plot(forecast['additive_terms'], label = 'Prophet')
plt.plot(output['seasonality'].reset_index(drop = True), label = 'Lazy')
plt.legend()
plt.show()
```
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_mean.png?raw=true "Output 1")
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_mean_trend.png?raw=true "Output 1")
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_mean_seasonality.png?raw=true "Output 1")


Toy Example: What is the potential impact of the coronavirus?
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
Some simulated data:
```python
import quandl
import fbprophet
import pandas as pd
import LazyProphet as lp

N = 730
t = np.linspace(0, 4*np.pi, N)
sine = 3.0*np.cos(t+0.001) + 0.5 + np.random.randn(N)
y = pd.Series(sine)
#some datetime index
y.index = pd.date_range(start=None, end='2020-04-05', periods=N)
df = pd.DataFrame(y, columns = ['y'])
df['ds'] = y.index
#fit prophet
model = fbprophet.Prophet(yearly_seasonality = True)
model.fit(df)
forecast = model.predict(df)
#%%
#create Lazy Prophet class
boosted_model = lp.LazyProphet(freq = 365, 
                            approximate_splits = True,
                            )
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
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_simulated_output.png?raw=true "Output")
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_simulated_trend.png?raw=true "Trends")
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_simulated_seasonality.png?raw=true "Seasonality")

# Plotting the components:
```python
import quandl
import pandas as pd
import matplotlib.pyplot as plt
import LazyProphet as lp

#Get bitcoin data
data = quandl.get("BITSTAMP/USD")
y = data['Low']
y = y[-730:]

#create Lazy Prophet class
boosted_model = lp.LazyProphet(freq = 365, 
                            estimator = 'linear', 
                            max_boosting_rounds = 50,
                            approximate_splits = True,
                            regularization = 1.2)
#Fits on just the time series
#returns a dictionary with the decomposition
output = boosted_model.fit(y)
boosted_model.plot_components()
```
![alt text](https://github.com/tblume1992/LazyProphet/blob/master/static/lazy_plot_components.png?raw=true "Output")
# Dealing with Exogenous Variables
Now let's take a look at exogenous variables which may have an effect on the BTC price. This is meant to be a demonstration using readily available information, the variables we use are just what comes with the Quandl request. 

Exogenous variables are fit in the last step of the boosting loop and all coefficients and standard errors are updated using all boosting rounds so the coefficients most likely are regularized.

Adding extra variables may also make the model want MORE boosting rounds, so we will increase the max_boosting_rounds.
```python
import quandl
import pandas as pd
import matplotlib.pyplot as plt
import LazyProphet as lp

#Get bitcoin data
data = quandl.get("BITSTAMP/USD")
#let's get our X matrix with the new variables to use
X = data.drop('Low', axis = 1)
X = X.iloc[-730:,:]
y = data['Low']
y = y[-730:]

#create Lazy Prophet class
boosted_model = lp.LazyProphet(freq = 365, 
                            estimator = 'linear', 
                            max_boosting_rounds = 200,
                            approximate_splits = True,
                            regularization = 1.2,
                            exogenous = X)
#Fits on just the time series
#returns a dictionary with the decomposition
output = boosted_model.fit(y)
boosted_model.summary()
```
The output is printed to the console, but all values also exist in the output dictionary from the fit() function.
```
***************Exogenous Model Results***************

        Coefficients  Standard Error  t-Stat  P-Value
High           -0.27            0.37   -0.74    0.460
Last            0.17           11.30    0.01    0.988
Bid             1.76           13.88    0.13    0.899
Ask            -2.09           14.19   -0.15    0.883
Volume         -0.02            0.01   -1.80    0.073
VWAP            1.11            0.51    2.16    0.031
```
# Forecasting
If you have no other variables and the problem is a simple Time Series setup then forecasting is just extrapolating the current measure of trend and seasonality utilizing the extrapolate(n_steps, future_X = None) method where n_steps is the number of steps to forecast and future_X is a dataframe/array for the future values of exogenous variables if you fit the model with any.  This just returns a numpy array not a series so beware!
```python
import quandl
import pandas as pd
import matplotlib.pyplot as plt
import LazyProphet as lp

#Get bitcoin data
data = quandl.get("BITSTAMP/USD")
#let's get our X matrix with the new variables to use
X = data.drop('Low', axis = 1)
X = X.iloc[-730:,:]
y = data['Low']
y = y[-730:]

#create Lazy Prophet class
boosted_model = lp.LazyProphet(freq = 365, 
                            estimator = 'linear', 
                            max_boosting_rounds = 200,
                            approximate_splits = True,
                            regularization = 1.2,
                            exogenous = X)
#Fits on just the time series
#returns a dictionary with the decomposition
output = boosted_model.fit(y)
forecast = boosted_model.extrapolate(30)
```
Many times we are not sure if the current trend will hold and would like the trend to be dampened over the forecast horizon to have a 0 slope, this can be done with the trend_dampening argument when building the class.  For this metric- a .5 would mean that the trend hits roughly half the value of the unconstrained trend by the end of the forecast horizon.  A .1 would mean the trend would hit roughly 90% of it's unconstrained value.  The dampenening is achieved via exponential decay of the slope and is a smooth transition for all involved.
```python
import quandl
import pandas as pd
import matplotlib.pyplot as plt
import LazyProphet as lp

#Get bitcoin data
data = quandl.get("BITSTAMP/USD")
#let's get our X matrix with the new variables to use
X = data.drop('Low', axis = 1)
X = X.iloc[-730:,:]
y = data['Low']
y = y[-730:]

#create Lazy Prophet class
boosted_model = lp.LazyProphet(freq = 365, 
                            estimator = 'linear', 
                            max_boosting_rounds = 200,
                            approximate_splits = True,
                            regularization = 1.2,
                            exogenous = X,
                            trend_dampening = .5)
#Fits on just the time series
#returns a dictionary with the decomposition
output = boosted_model.fit(y)
forecast = boosted_model.extrapolate(30)
```
