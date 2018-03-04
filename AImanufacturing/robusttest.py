import numpy as np
import pandas as pd 
from matplotlib import pyplot
from sklearn.preprocessing import robust_scale

#datatrain=pd.read_csv('../input/train.csv',header=0,index_col=0)

data=pd.read_csv('./input/train.csv')
print(data)
values=data#data.values

def num_missing(x):
    return sum(x.isnull())
def maxMinMedian(x):
    if not isinstance(x,str):
        return max(x),min(x)
def str_to_int(x):
	if isinstance(x,str):
		if x[1]:
			x=ord(x[0])+ord(x[1])
		if x[2]:
			x=ord(x[0])+ord(x[1])+ord(x[2])
		if x[3]:
			x=ord(x[0])+ord(x[1])+ord(x[2])+ord(x[3])
		x=ord(x)


data_all_int=(data.apply(str_to_int,axis=0)) # all data str to int  

print(data_all_int.head())

'''


pyplot.figure()
groups=[8,9]
i=1

for group in groups:
	pyplot.subplot(len(groups),1,i)
	pyplot.plot(values[:,group])
	pyplot.title(datatrain.columns[group],y=0.5,loc='right')
	i += 1
pyplot.show()


values_scale=..
pyplot.figure()
groups=[8,9]
i=1

for group in groups:
	pyplot.subplot(len(groups),1,i)
	pyplot.plot(values[:,group])
	pyplot.title(datatrain.columns[group],y=0.5,loc='right')
	i += 1
pyplot.show()

'''











