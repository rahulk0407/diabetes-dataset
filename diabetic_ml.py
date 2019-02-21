# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:39:00 2019

@author: rahul
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
  
from sklearn import datasets
dia= datasets.load_diabetes()
x=dia.data
y=dia.target
x=pd.DataFrame(x)
dia.feature_names
x.columns=dia.feature_names
y=pd.DataFrame(y)
y.columns=['result']
dia.DESCR
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.25, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

import statsmodels.formula.api as sm
x=np.append(arr=np.ones((442,1)).astype(int),values = x, axis = 1)


x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,2,3,4,5,6,7,8,9,10]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,2,3,4,5,6,8,9,10]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,2,3,4,5,6,8,9]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x_opt=x[:,[0,2,3,4,5,6,9]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()
x=x[:,[0,2,3,4,5,6,9]]
x=pd.DataFrame(x)
x=x.iloc[:,1:]
x.columns=['sex','bmi','bp','s1','s2','s5']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size= 0.25, random_state=0)
from sklearn.linear_model import LinearRegression
regressor2=LinearRegression()
regressor2.fit(x_train,y_train)
y_pred2=regressor2.predict(x_test)