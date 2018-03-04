

from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from pandas import Series,DataFrame
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


import resample,loadData_72mat

data = loadData_72mat.loadDataSet('A2done.csv')

##### 以8：00：00至8：20：00为例   ################################
data800=data[:,21]                       #所有日期8:00的数据为第21列
#tt=Series(data800,index=np.arange(90))   #转化为series
tt= Series(data800,index=pd.date_range('20/7/2016',periods=90))  ##转化为时间series
'''
ts=resample.A2done
ts.values[ts.values>300]=300  #去掉异常值
tt=ts['2016-9-1':'2016-9-5']
'''
tt.plot(figsize=(12,8))
plt.show()

################################## 判断差分阶数
'''
###### 一阶差分
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(111)
diff1 = tt.diff(1)
diff1.plot(ax=ax1)
plt.show()
'''
'''
###### 二阶差分
fig = plt.figure(figsize=(12,8))
ax2 = fig.add_subplot(111)
diff2 = tt.diff(2)
diff2.plot(ax=ax2)
plt.show()
'''

'''
tt = tt.diff(1)#我们已经知道要使用一阶差分的时间序列，之前判断差分的程序可以注释掉
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(tt,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(tt,lags=40,ax=ax2)
plt.show()
'''


## 画图AC和PAC判断p，q值
#tt = tt.diff(2)                 #差分效果不一定好
def draw_acf_pacf(tt, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(tt, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(tt, lags=31, ax=ax2)
    plt.show()

draw_acf_pacf(tt,lags=31)


'''
## 判断最佳模型
# AIC=-2 ln(L) + 2 k 中文名字：赤池信息量 akaike information criterion 
# BIC=-2 ln(L) + ln(n)*k 中文名字：贝叶斯信息量 bayesian information criterion 
# HQ=-2 ln(L) + ln(ln(n))*k hannan-quinn criterion 

## ARMA(p,q)的aic，bic，hqic均最小，是最佳模型
arma_mod20 = sm.tsa.ARMA(tt,(3,3)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
arma_mod30 = sm.tsa.ARMA(tt,(2,2)).fit()
print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
arma_mod40 = sm.tsa.ARMA(tt,(1,3)).fit()
print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
arma_mod50 = sm.tsa.ARMA(tt,(1,1)).fit()
print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)

'''
'''
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
'''




