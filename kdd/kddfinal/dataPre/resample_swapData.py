


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from pandas import Series,DataFrame
from datetime import datetime,date,time
from numpy.matlib import randn


def resample_CSV(route_id):
	simple=pd.read_csv('test2_20min_avg_travel_time_swapTest.csv')
	A2=simple[simple['route_id']==route_id]
	
	A2series=Series(A2['avg_travel_time'].values,index=A2['time_window_start'])

	A2series.index = pd.to_datetime(A2series.index,format='%Y-%m-%d %H:%M:%S')
	
	A2resample=A2series.resample('20T').mean()
	
	mm=A2resample.median()               
	A2full=A2resample.fillna(mm)
	
	A2done=A2full#[A2full.index>='2016-07-20 00:00:00']

	#A2done=Series(A2done.values,columns=['a'],index=A2done.index)
	print(route_id)
	print(A2done)
	A2done.to_csv(route_id.split('-')[0]+route_id.split('-')[1]+'_resample.csv')












'''
#
simple=pd.read_csv('training_20min_avg_travel_time_simple.csv')

type(simple)
#pandas.core.frame.DataFrame

simple.index
#RangeIndex(start=0, stop=25144, step=1)

simple.columns
#Index(['route_id', 'time_window_start', 'avg_travel_time'], dtype='object')

A2=simple[simple['route_id']=='A-2']
#注意A2是从DataFrame中选取的列，它是一个Series，取值必须去values！！！   [].values  [].index
A2series=Series(A2['avg_travel_time'].values,index=A2['time_window_start'])

#In [20]: A2resample=A2series.resample('20T').mean()
#TypeError: Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of 'Index'

#index类型是obj，不能resamp，必须转化为datetime类型！！！！
A2series.index = pd.to_datetime(A2series.index,format='%Y-%m-%d %H:%M:%S')

#resample成功！！！
A2resample=A2series.resample('20T').mean()

#取A2resample的values中位数填充NaN值
mm=A2resample.median()               
A2full=A2resample.fillna(mm)

#去掉开头不对齐数据，从20号开始计时
A2done=A2full[A2full.index>='2016-07-20 00:00:00']
#job done !!!
print('A2done 如下：')
print(A2done)
#导出CSV


A2done.to_csv('A2done0517.csv')
'''
'''
#从文件读出csv，并将string拆分为列表
num=open('A2done.csv').readline().strip().split(',')
#原格式为string：'2016-07-20 00:00:00,10.6\n'
#In [61]: num
#Out[61]: ['2016-07-20 00:00:00', '10.6']

################# 将数据读成每天一条记录，20min一条数据，数据为avg time   #############
######  数据转化为ndarry类型   #########################################
fr = open('A2done.csv')
dataMat=[]
lineArr = []
for line in fr.readlines():
	curLine = line.strip().split(',')  #列表出现
	lineArr.append(float(curLine[1]))  #append单个元素, curLine[1] is data of avg time
	if len(lineArr) == 24*3:           #每天72条数据     
		dataMat.append(lineArr)
		lineArr=[]


#########################    dataMat是list类型
#### nddata为ndarray类型 ########################
nddata = np.array(dataMat)       ################## ndata为72列 #########

print(nddata[0,:])      #####  共72列，从0：00：00 至 23：40：00  


xMat = np.mat( nddata[:,15:21] )   #### 最后一个值不取 #########

yMat = np.mat( nddata[:,21] ).T 


'''



