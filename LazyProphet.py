import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge



class LazyProphet():
  def __init__(self, 
               freq = 0, 
               estimator = 'linear', 
               max_boosting_rounds = 50, 
               l2 = 1, 
               poly = 3, 
               nested_seasonality = False, 
               ols_constant = False,
               seasonal_smoothing = 15,
               approximate_splits = False,
               regularization = 1.0):
    
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
    
  def ridge(self,y):
    if len(y) == 1:
      predicted = np.array(y[0])
    else:
      y = np.array(y).reshape((len(y), 1)) 
      X = np.array(list(range(len(y))), ndmin=1).reshape((len(y), 1))   
      X = PolynomialFeatures(degree = self.poly, include_bias = False).fit(X).transform(X) 
      clf = Ridge(alpha=self.l2).fit(X, y)
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
        iteration_mae = np.mean(np.abs(y - np.append(predicted1,predicted2)))
        if index == 0:
          mae = iteration_mae
        if iteration_mae <= mae:
          mae = iteration_mae  
          predicted = np.append(predicted1,predicted2)
      
    return predicted

  def ols(self,y, bias, ols_constant = False):
    y = np.array(y - bias).reshape(-1, 1)
    X = np.array(range(len(y))).reshape(-1, 1)
    if ols_constant:
      X = np.append(X, np.asarray(np.ones(len(y))).reshape(len(y), 1), axis = 1)
    beta =  np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
    predicted = X.dot(beta) + bias
    
    return predicted

  def linear(self,y):
    if len(y) == 1:
      predicted = np.array(y[0])
    else:
      proposals = self.get_split_proposals()
      for index, i in enumerate(proposals): 
        predicted1 = self.ols(y[:i], 0, ols_constant = True)
        predicted2 = self.ols(y[i:], predicted1[-1], ols_constant = False)
        iteration_mae = np.mean(np.abs(y - np.append(predicted1,predicted2)))
        if index == 0:
          mae = iteration_mae
        if iteration_mae <= mae:
          mae = iteration_mae  
          predicted = np.append(predicted1,predicted2)
      
    return predicted

  def get_approximate_split_proposals(self):
    gradient = pd.Series(np.abs(np.gradient(self.boosted_data, edge_order=1)))
    gradient = gradient.sort_values(ascending = False)[:100]
    proposals = list(gradient.index)
    proposals = [i for i in proposals if i > 10]
    proposals = [i for i in proposals if i < len(self.boosted_data) - 10]
    
    return proposals
    
    
  def get_split_proposals(self):
    if self.approximate_splits:
      return self.get_approximate_split_proposals()
    else:
      return list(range(10, len(self.boosted_data) - 10))
  
  
  def get_seasonality(self,y,freq):
    if not self.freq:
      seasonality = np.zeros(len(self.time_series))
    else:
      seasonality = np.array([np.mean(y[i::freq], axis=0) for i in range(freq)])
      if freq > 30:
        b, a = scipy.signal.butter(20, 0.125)
        seasonality = scipy.signal.filtfilt(b,a,seasonality)
      
    return seasonality
  
  def filtered_seasonality(self,total_seasonality):
    if not self.freq:
      filtered_sig = np.zeros(len(self.time_series))
    else:
      total_seasonality = np.append(total_seasonality[:self.freq],total_seasonality[0])
      N = len(total_seasonality)
      ft = np.fft.fft(total_seasonality - np.mean(total_seasonality))/N
      ft[ft.argsort()[self.seasonal_smoothing:]] = 0
      #welch = scipy.signal.welch(y) 
      filtered_sig = (scipy.fftpack.ifft(ft+ np.mean(total_seasonality)))*N 
      filtered_sig += np.mean(total_seasonality)
      filtered_sig = filtered_sig[1:].real
      filtered_sig = np.resize(np.array(filtered_sig)[:self.freq + 2], len(self.time_series))
    
    return filtered_sig
  
  def calc_cost(self, prediction, c):
    n = len(self.time_series)
    return n*np.log(np.sum((self.time_series - prediction )**2)/n) + ((c)**self.regularization) * np.log(n)
  
  def fit(self, time_series):
    self.time_series_index = time_series.index
    self.time_series = time_series.values
    trends = []
    seasonalities = []
    nested_seasonalities = []
    errors =[]
    self.boosted_data = self.time_series
    for i in range(self.max_boosting_rounds):
      if self.estimator == 'ridge':
        trend = self.ridge(self.boosted_data).reshape(self.boosted_data.size,)
      elif self.estimator == 'mean':
        if i == 0:
          trend = np.tile(np.mean(self.boosted_data), len(self.boosted_data))
        else:
          trend = self.mean(self.boosted_data).reshape(self.boosted_data.size,)
      elif self.estimator == 'linear':
        if i == 0:
          trend = np.tile(np.mean(self.boosted_data), len(self.boosted_data))
        else:
          trend = self.linear(self.boosted_data).reshape(self.boosted_data.size,)
      resid = self.boosted_data-trend
      seasonality = self.get_seasonality(resid, self.freq)
      if self.freq:
        seasonality = np.tile(seasonality,
                              int(len(self.time_series)/self.freq + 1))[:len(self.time_series)]
      if self.nested_seasonality:
        nested_seasonal_factor = self.get_seasonality(self.boosted_data-(trend + seasonality), 7)
        nested_seasonal_factor =  np.tile(nested_seasonal_factor,
                                          int(len(self.time_series)/7 + 1))[:len(self.time_series)]
        nested_seasonalities.append(nested_seasonal_factor)
        self.boosted_data = self.boosted_data-(trend+seasonality+nested_seasonal_factor)
      else:
        self.boosted_data = self.boosted_data-(trend+seasonality)
      
      errors.append(np.mean(np.abs(self.boosted_data)))
      trends.append(trend)
      seasonalities.append(seasonality)
      total_trend = np.sum(trends, axis = 0)
      total_seasonalities = np.sum(seasonalities, axis = 0)
      round_cost = self.calc_cost(total_trend + total_seasonalities, i)
      if i == 0:
        cost = round_cost
      if round_cost <= cost:
        cost = self.calc_cost(total_trend + total_seasonalities, i)
      else:
        break
    total_seasonalities = self.filtered_seasonality(total_seasonalities)
    if self.nested_seasonality:
      total_nested_seasonality = np.sum(nested_seasonalities, axis = 0)
    output = {}
    output['y'] = self.time_series
    if self.nested_seasonality:
      output['yhat'] = pd.Series(total_trend + 
                                total_seasonalities + 
                                total_nested_seasonality, 
                                index = self.time_series_index).astype(float)
    else:
      output['yhat'] = pd.Series(total_trend + 
                                total_seasonalities, 
                                index = self.time_series_index).astype(float)
    output['trend'] = pd.Series(total_trend, index = self.time_series_index)
    output['seasonality'] = pd.Series(total_seasonalities, 
                                      index = self.time_series_index)
    if self.nested_seasonality:
      output['nested_seasonality'] = pd.Series(total_nested_seasonality, 
                                                index = self.time_series_index)
      
    return output
