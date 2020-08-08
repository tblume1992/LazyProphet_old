import pandas as pd
import numpy as np
from scipy import signal
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import statsmodels.api as sm
import matplotlib.pyplot as plt

class LazyProphet:
  def __init__(self, 
               freq = 0, 
               estimator = 'linear', 
               max_boosting_rounds = 100, 
               l2 = .01, 
               poly = 3, 
               nested_seasonality = False, 
               ols_constant = False,
               seasonal_smoothing = 10,
               approximate_splits = True,
               regularization = 1.2,
               seasonal_esti = 'harmonic',
               split_cost = 'mse',
               global_cost = 'maicc',
               trend_dampening = 0,
               seasonal_regularization = 'auto',   
               exogenous = None,
               verbose = 0,
               n_split_proposals = 100,
               min_samples = .15,
               exclude_splits = [],
               min_boosting_rounds = 0,
               exogenous_fit = 'ols'
               ):
    
    self.l2 = l2
    self.nested_seasonality = nested_seasonality
    self.poly = poly
    self.freq = freq
    self.max_boosting_rounds = max_boosting_rounds
    self.estimator = estimator
    self.ols_constant = ols_constant
    self.seasonal_smoothing = seasonal_smoothing
    self.approximate_splits = approximate_splits
    self.regularization = regularization
    self.seasonal_esti = seasonal_esti
    self.split_cost = split_cost
    self.global_cost = global_cost
    self.trend_dampening = trend_dampening
    self.exogenous = exogenous
    self.verbose = verbose
    self.n_split_proposals = n_split_proposals
    self.min_samples = min_samples
    self.exclude_splits = exclude_splits
    self.min_boosting_rounds = min_boosting_rounds
    self.exogenous_fit = exogenous_fit
    if type(self.freq) != int:
        raise Exception('Frequency must be an int')
    
    if seasonal_regularization != 'auto':
        self.seasonal_regularization = 1 - seasonal_regularization 
    else:
        self.seasonal_regularization = seasonal_regularization
    self.coefs = np.ones((1,poly))
    
  def ridge(self,y):
    if len(y) == 1:
      predicted = np.array(y[0])
    else:
      y = np.array(y).reshape((len(y), 1)) 
      X = np.array(list(range(len(y))), ndmin=1).reshape((len(y), 1))   
      X = PolynomialFeatures(degree = self.poly, include_bias = False).fit(X).transform(X) 
      clf = Ridge(alpha=self.l2).fit(X, y)
      self.coefs += clf.coef_
      predicted = clf.predict(X)  
      
    return predicted
  
  def mean(self,y):
    if len(y) == 1:
      predicted = np.array(y[0])
    else:
      proposals = self.get_split_proposals()
      for index, i in enumerate(proposals):  
        predicted1 = np.tile(np.mean(y[:i]), len(y[:i]))
        predicted2 = np.tile(np.mean(y[i:]), len(y[i:]))
        iteration_mae = self.get_split_cost(y, predicted1, predicted2)
        if index == 0:
          mae = iteration_mae
        if iteration_mae <= mae:
          mae = iteration_mae  
          predicted = np.append(predicted1,predicted2)
      
    return predicted
  
  def ols(self, y, bias, ols_constant):
    y = np.array(y - bias).reshape(-1, 1)
    X = np.array(range(len(y))).reshape(-1, 1)
    if ols_constant:
      X = np.append(X, np.asarray(np.ones(len(y))).reshape(len(y), 1), axis = 1)
    beta =  np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
    predicted = X.dot(beta) + bias
    
    return predicted

  def get_split_cost(self, y, split1, split2):
    if self.split_cost == 'mse':
      cost = np.mean((y - np.append(split1,split2))**2)
    elif self.split_cost == 'mae':
      cost = np.mean(np.abs(y - np.append(split1,split2)))
      
    return cost

  def linear(self,y):
    if len(y) == 1:
      predicted = np.array(y[0])
    else:
      proposals = self.get_split_proposals()
      for index, i in enumerate(proposals): 
        predicted1 = self.ols(y[:i], 0, ols_constant = True)
        predicted2 = self.ols(y[i:], predicted1[-1], ols_constant = self.ols_constant)
        iteration_mae = self.get_split_cost(y, predicted1, predicted2)
        if index == 0:
          mae = iteration_mae
        if iteration_mae <= mae:
          mae = iteration_mae  
          predicted = np.append(predicted1,predicted2)
      
    return predicted

  def get_approximate_split_proposals(self):
    gradient = pd.Series(np.abs(np.gradient(self.boosted_data, edge_order=1)))
    gradient = gradient.sort_values(ascending = False)[:self.n_split_proposals]
    proposals = list(gradient.index)
    min_split_idx = int(len(self.boosted_data)*.1)
    proposals = [i for i in proposals if i > min_split_idx]
    proposals = [i for i in proposals if i < len(self.boosted_data) - min_split_idx]
    proposals = [i for i in proposals if i not in self.exclude_splits]

    return proposals

  def get_fourier_series(self, t, p=365, n=10):
    x = 2 * np.pi * np.arange(1, n + 1) / p
    x = x * t[:, None]
    self.fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    
    return    
    
  def get_split_proposals(self):
    if self.approximate_splits:
      return self.get_approximate_split_proposals()
    else:
      min_idx =int(max(5, len(self.time_series) * self.min_samples))
      return list(range(min_idx, len(self.boosted_data) - min_idx))
  
  
  def get_harmonic_seasonality(self,y,freq):
    if not self.freq:
      seasonality = np.zeros(len(self.time_series))
    else:
      X = self.fourier_series
      X = np.append(X, np.asarray(np.ones(len(y))).reshape(len(y), 1), axis = 1)
      beta =  np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
      seasonality = X @ beta
      if self.seasonal_regularization == 'auto':
          seasonal_reg = seasonality.reshape((len(seasonality), 1))
          regularization = float(np.linalg.pinv(seasonal_reg.T.dot(seasonal_reg)).dot(seasonal_reg.T.dot(self.boosted_data)))
          if regularization > 1:
              regularization = 1
          if regularization < 0:
              regularization = 0
          seasonality = seasonality * regularization
      else:
          seasonality = seasonality * self.seasonal_regularization
    return seasonality 

  def get_naive_seasonality(self,y,freq):
    if not self.freq:
      seasonality = np.zeros(len(self.time_series))
    else:
      seasonality = np.array([np.mean(y[i::freq], axis=0) for i in range(freq)])
      if freq > 30:
        b, a = signal.butter(20, 0.125)
        seasonality = signal.filtfilt(b,a,seasonality)
      
    return seasonality * self.seasonal_regularization
  
  def calc_cost(self, prediction, c):
    n = len(self.time_series)
    if self.global_cost == 'maic':
      cost = 2*(c**self.regularization) + n*np.log(np.sum((self.time_series - prediction )**2)/n)
    if self.global_cost == 'maicc':
      cost = (2*c**2 + 2*c)/(n-c-1) + 2*(c**self.regularization) + \
              n*np.log(np.sum((self.time_series - prediction )**2)/n)    
    elif self.global_cost == 'mbic':
      cost = n*np.log(np.sum((self.time_series - prediction )**2)/n) + \
            (c**self.regularization) * np.log(n)
    return cost

  def get_trend(self, time_series, iteration):
    if self.estimator == 'ridge':
      trend = self.ridge(self.boosted_data).reshape(self.boosted_data.size,)
    elif self.estimator == 'mean':
      if iteration == 0:
        trend = np.tile(np.median(self.boosted_data), len(self.boosted_data))
      else:
        trend = self.mean(self.boosted_data).reshape(self.boosted_data.size,)
    elif self.estimator == 'linear':
      if iteration == 0:
        trend = np.tile(np.median(self.boosted_data), len(self.boosted_data))
        #trend = self.ols(self.boosted_data, 0, True).reshape(self.boosted_data.size,)
      else:
        trend = self.linear(self.boosted_data).reshape(self.boosted_data.size,)
        
    return trend

  def get_seasonality(self, detrended):
    if self.seasonal_esti == 'harmonic':
      seasonality = self.get_harmonic_seasonality(detrended, self.freq)
    elif self.seasonal_esti == 'naive':
      seasonality = self.get_naive_seasonality(detrended, self.freq)
    seasonality = np.tile(seasonality,
                            int(len(self.time_series)/self.freq + 1))[:len(self.time_series)]
    return seasonality
  
  def fit(self, time_series):
    self.time_series_index = time_series.index
    self.time_series = time_series.values
    if self.seasonal_esti == 'harmonic' and self.freq:
      self.get_fourier_series(np.arange(len(self.time_series)), self.freq, self.seasonal_smoothing)
    trends = []
    seasonalities = []
    nested_seasonalities = []
    errors =[]
    exo_predicted = []
    exo_impact = []
    self.boosted_data = self.time_series
    for i in range(self.max_boosting_rounds):
      trend = self.get_trend(self.boosted_data, iteration = i)
      resid = self.boosted_data-trend
      if self.freq:
          seasonality = self.get_seasonality(detrended = resid)
      else:
          seasonality = np.zeros(len(resid))
      if self.nested_seasonality:
        nested_seasonal_factor = self.get_naive_seasonality(self.boosted_data-(trend + seasonality), 7)
        nested_seasonal_factor =  np.tile(nested_seasonal_factor,
                                          int(len(self.time_series)/7 + 1))[:len(self.time_series)]
        nested_seasonalities.append(nested_seasonal_factor)
        self.boosted_data = self.boosted_data-(trend+seasonality+nested_seasonal_factor)
      else:
        self.boosted_data = self.boosted_data-(trend+seasonality)  
      if self.exogenous is not None:
          if self.exogenous_fit == 'ols':
              exo_model = sm.OLS(self.boosted_data, self.exogenous).fit()
              exo_predict = exo_model.predict(self.exogenous)
          elif self.exogenous_fit == 'glm':
              exo_model = sm.GLM(self.boosted_data, self.exogenous).fit()
              exo_predict = exo_model.predict(self.exogenous)
          self.boosted_data = self.boosted_data - exo_predict
          exo_impact.append((exo_model.params, exo_model.cov_params()))
          exo_predicted.append(exo_predict)
      errors.append(np.mean(np.abs(self.boosted_data)))
      trends.append(trend)
      seasonalities.append(seasonality)
      total_trend = np.sum(trends, axis = 0)
      total_seasonalities = np.sum(seasonalities, axis = 0)
      total_exo = np.sum(exo_predicted, axis = 0)
      #Get a measure of complexity: number of splits + any extra variables
      if self.freq != 0:
        c = i + self.seasonal_smoothing + 1
      else:
        c = i + 1
      if self.estimator == 'ridge':
        c += self.poly
      if self.exogenous is not None:
          c += np.shape(self.exogenous)[1]
      round_cost = self.calc_cost(total_trend + total_seasonalities + total_exo, c)
      if i == 0:
        cost = round_cost
      if round_cost <= cost or i < self.min_boosting_rounds:
        cost = round_cost
        self.total_trend = total_trend
        self.total_seasonalities = total_seasonalities
        self.total_exo = total_exo
      else:
        break

    if self.nested_seasonality:
      total_nested_seasonality = np.sum(nested_seasonalities, axis = 0)
    output = {}
    output['y'] = pd.Series(self.time_series, index = self.time_series_index)
    if self.nested_seasonality:
      yhat = pd.Series(self.total_trend + 
                                self.total_seasonalities + 
                                total_nested_seasonality + self.total_exo, 
                                index = self.time_series_index).astype(float)
    else:
      yhat = pd.Series(self.total_trend + 
                                self.total_seasonalities + self.total_exo, 
                                index = self.time_series_index).astype(float)
    trend = pd.Series(self.total_trend, index = self.time_series_index)
    seasonality = pd.Series(self.total_seasonalities, 
                                      index = self.time_series_index)
    exogenous = pd.Series(self.total_exo, index = self.time_series_index)
    if self.nested_seasonality:
      output['nested_seasonality'] = pd.Series(total_nested_seasonality, 
                                                index = self.time_series_index)
    self.seasonal_strength = self.calc_seasonal_strength(self.time_series - yhat,
                                      self.time_series - trend)
    if self.seasonal_strength > 0 and self.seasonal_strength <= .15 and self.verbose:
        print('Seasonal Signal is weak, try a different frequency or disable seasonality with freq=0')
    self.trend_strength = self.calc_trend_strength(self.time_series - yhat,
                                      self.time_series - seasonality)
    
    
    upper_prediction, lower_prediction = self.get_prediction_intervals(self.time_series,
                                                                       yhat)
    if self.exogenous is not None:
        output['Exogenous Prediction'] = exogenous
        output['Exogenous Summary'] = self.get_boosted_exo_results(exo_impact)
        self.exo_impact = exo_impact
    if self.freq != 0:
        c = i + self.seasonal_smoothing + 1
    else:
        c = i + 1
    if self.estimator == 'ridge':
        c += self.poly
    n = len(self.time_series)
    
    self.model_cost = (2*c**2 + 2*c)/(n-c-1) + 2*(c**self.regularization) + \
              n*np.log(np.sum((self.time_series - yhat )**2)/n) 
    output['yhat'] = yhat
    output['yhat_upper'] = upper_prediction
    output['yhat_lower'] = lower_prediction
    output['seasonality'] = seasonality
    output['trend'] = trend
    self.output = output
    self.number_of_rounds = i
    
    return output

  def get_prediction_intervals(self, y, predicted):
    sd_error = np.std(y - predicted)
    t_stat = stats.t.ppf(.9, len(y))
    upper = predicted + t_stat*sd_error
    lower = predicted - t_stat*sd_error
    
    return upper, lower

  def calc_seasonal_strength(self, resids, detrended):
    return max(0, 1-(np.var(resids)/np.var(detrended)))

  def calc_trend_strength(self, resids, deseasonalized):
    return max(0, 1-(np.var(resids)/np.var(deseasonalized)))

  def extrapolate(self, n_steps = 10, future_X = None):
    if self.estimator == 'mean':
      extrapolated_trend = np.tile(self.total_trend[-1], n_steps)
    else:
      extrapolated_trend = (self.total_trend[-1] - self.total_trend[-2])* \
                            np.arange(1,n_steps + 1) + \
                            self.total_trend[-1]
    extrapolated_seasonality = np.resize(self.total_seasonalities[:self.freq], 
                                         len(self.total_seasonalities) + n_steps)[-n_steps:]
    
    if self.trend_dampening:
        extrapolated_trend = self.trend_dampen(self.trend_dampening, extrapolated_trend)
    extrapolated_series = extrapolated_trend + extrapolated_seasonality
    if future_X is not None:
        extrapolated_series += np.dot(future_X,np.array(self.beta).reshape(-1,1)).reshape(-1,)
    
    return extrapolated_series

  def trend_dampen(self, damp_fact, trend):
     zeroed_trend = trend - trend[0]
     damp_fact = 1 - damp_fact
     if damp_fact < 0:
         damp_fact = 0
     if damp_fact > 1:
         damp_fact = 1
     if damp_fact == 1:
         dampened_trend = zeroed_trend
     else:       
         tau = (damp_fact*1.15+(1.15*damp_fact/.85)**9)*\
                 (2*len(zeroed_trend))
         dampened_trend = (zeroed_trend*np.exp(-pd.Series(range(1, len(zeroed_trend) + 1))/(tau)))
         crossing = np.where(np.diff(np.sign(np.gradient(dampened_trend))))[0]
         if crossing.size > 0:
             crossing_point = crossing[0]
             dampened_trend[crossing_point:] = dampened_trend[(np.mean(np.gradient(zeroed_trend))*dampened_trend).idxmax()]
   
     return dampened_trend + trend[0]
 
  def get_boosted_exo_results(self, exo_impacts):
    y = self.time_series
    for i, element in enumerate(exo_impacts[:-1]):
        if i == 0:
            var = np.diag(element[1])
            beta = element[0]
        else:
            var = ((len(y) - 1) / len(y))**2 * var +(1/len(y)**2)*(len(y) - 1)*np.diag(element[1])
            beta += element[0]
    se = np.sqrt(var)
    t_stat = beta/se
    pval = stats.t.sf(np.abs(t_stat), len(y)-1)*2
    self.var = var
    self.beta = beta
    summary = {}
    summary['P-Value'] = pval
    summary['t-Stat'] = t_stat
    summary['Coefficient'] = beta
    summary['Standard Error'] = se
    
    return summary

  def plot_components(self):
      summary_dict = self.output
      if 'Exogenous Prediction' in summary_dict.keys():
          fig, ax = plt.subplots(4, figsize = (8,8))
          ax[-2].plot(summary_dict['Exogenous Prediction'], color = 'orange')
          ax[-2].set_title('Exogenous')
      else:
          fig, ax = plt.subplots(3, figsize = (8,8))
      ax[0].plot(summary_dict['trend'], color = 'orange')
      ax[0].set_title('Trend')
      ax[1].plot(summary_dict['seasonality'], color = 'orange')
      ax[1].set_title('Seasonality')
      ax[-1].plot(summary_dict['y'], color = 'black')
      ax[-1].plot(summary_dict['yhat'], color = 'orange')
      ax[-1].plot(summary_dict['yhat_upper'], 
                 linestyle = 'dashed', 
                 alpha = .5,
                 color = 'orange')
      ax[-1].plot(summary_dict['yhat_lower'],
                 linestyle = 'dashed', 
                 alpha = .5,
                 color = 'orange')
      ax[-1].set_title('Fitted')
      
      plt.show()
      
  def summary(self):
    try:
        exo_summary = self.output['Exogenous Summary']
    except:
        raise Exception('No exogenous variables to summarize')
    asterisks = '*'*15
    summary_dataframe = pd.DataFrame(exo_summary['Coefficient'].round(2), 
                                     columns = ['Coefficients'],
                                     index = exo_summary['Coefficient'].index)
    summary_dataframe['Standard Error'] = exo_summary['Standard Error'].round(2)
    summary_dataframe['t-Stat'] = exo_summary['t-Stat'].round(2)
    summary_dataframe['P-Value'] = exo_summary['P-Value'].round(3)
    print(f'''\n{asterisks}Exogenous Model Results{asterisks}\n''')
    print(summary_dataframe)
