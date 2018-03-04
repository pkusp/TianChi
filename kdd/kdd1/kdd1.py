import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from pandas import Series,DataFrame
from datetime import datetime,date,time
from numpy.matlib import randn
#import matplotlib.pyplot as plt
#a=pd.read_csv('A2_8_00.CSV')
#print(a)

print("sublime is open")


ccc={'8:00': {1: {0:0,1:0},2: {0:0,1:0},3:{0:0,1:0}},'8:20': {1: {0:0,1:0},2: {0:0,1:0},3:{0:0,1:0}}}
print(ccc)
ddd=DataFrame(ccc)
print(ddd)

 ccc={datetime(2011,1,2): {1: {0:0,1:0},2: {0:0,1:0},3:{0:0,1:0}},datetime(2011,1,3):{1: {0:0,1:0},2: {0:0,1:0},3:{0:0,1:0}}}
 print(ccc)
 ddd=DataFrame(ccc)
 print(ddd)


# 读取数据，pd.read_csv默认生成DataFrame对象，需将其转换成Series对象
df = pd.read_csv('A2done.csv') #, encoding='utf-8', index_col='date')

#df.index = pd.to_datetime(df.index)  # 将字符串索引转换成时间索引
#tt= Series(df,index=pd.date_range('20/7/2016',periods=90))  ##转化为时间series
################### 将A2done 的数据转化为 以时间为index的 series 格式数据  ###############################
ts = Series(df.values[:,1],index=pd.to_datetime(df.values[:,0]))   ####     第0列为时间窗口，第1列为行车时间  ############

ts = ts['2016-9-1':]


import resample,loadData_72mat

data = loadData_72mat.loadDataSet('A2done.csv')

##### 以8：00：00至8：20：00为例   ################################
data800=data[:,21]                       #所有日期8:00的数据为第21列
#tt=Series(data800,index=np.arange(90))   #转化为series
tt= Series(data800,index=pd.date_range('20/7/2016',periods=90))  ##转化为时间series













#####################  resample 脚本！！！   ################################
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

#取A2resample的values最小值填充NaN值
mm=min(A2resample.values);A2full=A2resample.fillna(mm)

#去掉开头不对齐数据，从20号开始计时
A2done=A2full[A2full.index>='2016-07-20 00:00:00']
#job done !!!
print(A2done)
#导出CSV
A2done.to_csv('A2done.csv')
#######################################  resample   #####3###########################
'''


'''   #######################################重采样技术！！！
dates = [datetime(2011,1,2),datetime(2011,1,5),datetime(2011,1,7,20,30,15)]

ts = Series([666,555,444],index=dates)

print(ts)
ts1 = ts.resample('20T').mean()
print(ts1)
##############################################################
 '''

'''################################################
        #root_time_window is a list!
        #插入初始值00：00：00
        if route_time_windows[0].hour != 0 or route_time_windows[0].minute != 0:
            temp = route_time_windows[0]
            temp.replace(hour = 0,minute = 0)
            temp = list(temp)
            route_time_windows = temp + route_time_windows 

        route_time_windows.resample('20T').mean()     #resample方法，绝对利器！！！
'''
'''  #不熟悉pandas在自己编函数，事倍功半！！！！！！只要一句resample就行了！！！
        i = route_time_windows[0]
        while i != route_time_windows[-1]:
            if route_time_windows[i+1] != route_time_windows[i] + timedelta(minutes=20):
                route_time_windows[]
#####################################################
'''








