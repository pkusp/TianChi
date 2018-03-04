import pandas as pd
import numpy as np 
from pandas import Series,DataFrame

def mergeMat(fileName):
	name1=fileName+'_72mat.csv'
	name2=fileName+'_swap__72mat.csv'
	temp1=pd.read_csv(name1)
	temp2=pd.read_csv(name2)

	temp01=temp1.drop('day',axis=1)
	temp02=temp2.drop('Unnamed: 0',axis=1)
	arr1=np.array(temp01)
	arr2=np.array(temp02)
	newArr=np.vstack((arr1,arr2))

	newDF=DataFrame(newArr)
	newDF.to_csv(fileName+'_72matTrain.csv')

route=['A2','A3','B1','B3','C1','C3']
for name in route:
	mergeMat(name)

