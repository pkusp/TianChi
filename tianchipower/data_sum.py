###############################################################
#
#  电力影响因素   经济数据（stock market） 天气 
#
#  可以先预测 电力的影响因素 当做新增的特征 来预测用电量
#
#  data_sum.py 完成日期 2017.06.02
#
##############################################################
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from pandas import Series,DataFrame

def dataSum(fileName):
	data_raw=pd.read_csv(fileName)
	type(data_raw)
	#In [19]: data_raw.columns
	#Out[19]: Index(['record_date', 'user_id', 'power_consumption'], dtype='object')
	column_date_str = data_raw['record_date']
	type(column_date_str)                                          ## pandas.core.series.Series
	column_date=pd.to_datetime(column_date_str,format='%Y-%m-%d')

	data_raw['record_date']=column_date                            ##  日期替换为时间格式

	date_list=[]  ## data_list 用来append每个日期的总电量
	for each_date in pd.date_range('2016/9/1','2016/9/30',freq='1D'): ## power汇总
		DF_each_date = data_raw[data_raw['record_date']==each_date]
		powerSum_each_date = DF_each_date['power_consumption'].sum()
		date_list.append(powerSum_each_date)  ##

	date_pre=Series(date_list,index=pd.date_range('2016/9/1','2016/9/30',freq='1D'))   # 日期和总电量两个 list 合并为 Series

	print(date_pre)   ##  总用电量 is here ！！
	date_pre.to_csv('powerSum_9.csv')

#dataSum('Tianchi_power.csv')  ## 导出加总后的csv文件，文件名为 ‘powerSum.csv’ 




