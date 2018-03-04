import numpy as np
import pandas as pd 
from matplotlib import pyplot
from sklearn.preprocessing import robust_scale

datatrain=pd.read_csv('../input/train.csv',header=0,index_col=0)
datatest=pd.read_csv('../input/testA.csv')
datascale=pd.read_csv('../input/train_OneHot_padFill_robustScale.csv',header=0,index_col=0)

train_arr=np.array(datatrain)
a=[[1,2,3,4,4,4,5,5,5,6,6,6,7,7,7,100],[1,2,3,4,4,4,5,5,5,6,6,6,7,7,7,1000]]
data=robust_scale(a)
print(data)
'''

values=data#data.values

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
