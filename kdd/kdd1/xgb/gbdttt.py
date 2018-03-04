

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from pandas import Series,DataFrame

gbdt=GradientBoostingRegressor(
  loss='ls'                        #均方误差
, learning_rate=0.05
, n_estimators=100
, subsample=0.6
#, #min_samples_split=None
#, #min_samples_leaf=None
#, #max_depth=None
, init=None
#, #random_state=None
#, #max_features=None
, alpha=0.9
, verbose=0
#, #max_leaf_nodes=None
, warm_start=False
)
'''
train_feat=np.genfromtxt("train_feat.txt",dtype=np.float32)
train_id=np.genfromtxt("train_id.txt",dtype=np.float32)
test_feat=np.genfromtxt("test_feat.txt",dtype=np.float32)
test_id=np.genfromtxt("test_id.txt",dtype=np.float32)
'''
def each_gbdt(fileName):
	
	train_name = fileName+'_72mat.csv'
	test_name = fileName + '_72mat01.csv'
	train = pd.read_csv(train_name)
	test = pd.read_csv(test_name)

	train=train.drop('Unnamed: 0',axis=1)
	test=test.drop('Unnamed: 0',axis=1)

	train=np.array(train)    #DataFrame to ndarray
	test=np.array(test)


	resultArr=[]
	#resultList=[]

	for i in range(6): 

		j = 24+i
		train_x=train[:,18:24]   #特征
		train_y=train[:,j]        #label 从第24列开始，共执行六次

		test_x=test[:,18:24]     #测试数据
		#test_id=test[24]

		train_x=np.mat(train_x)   # ndarray to mat
		test_x=np.mat(test_x)   # ndarray to mat
		#test_id=np.mat(test_id)
		train_y=np.mat(train_y).T    ## 列向量！！！！！！！！！！！！！！！！！！！！！！！！！

		#print(train_feat.shape)#,train_id.shape,test_feat.shape)#,test_id.shape)

		gbdt.fit(train_x,train_y)
		pred=gbdt.predict(test_x)

		resultArr.append(pred)     #

	resultMat=np.mat(resultArr)        # arr to mat
	#resultMat=resultMat.T           # .T

	print('this result is:    '+fileName+'   new morning data prediction!!')
	print(resultMat)
	print(resultMat.shape)
	print('\n')


	resultArr_2=[]
	for i in range(6):        #######  15:00 to 17:00 predict 17:00 to 19:00  ##########

		j = 51+i
		train_x=train[:,45:51]   #特征 15:00 to 17:00
		train_y=train[:,j]        #label 从第51列开始，共执行六次

		test_x=test[:,45:51]     #测试数据
		#test_id=test[24]

		train_x=np.mat(train_x)   # ndarray to mat
		test_y=np.mat(test_x)   # ndarray to mat
		#test_id=np.mat(test_id)
		train_y=np.mat(train_y).T    ## 列向量！！！！！！！！！！！！！！！！！！！！！！！！！

		print(train_x.shape,train_y.shape,test_x.shape)#,test_id.shape)

		gbdt.fit(train_x,train_y)
		pred=gbdt.predict(test_x)

		resultArr_2.append(pred)     #

	resultMat_2=np.mat(resultArr_2)        # arr to mat
	#resultMat_2=resultMat_2.T           # .T

	print('this result is:   '+fileName+'   new afternoon data prediction!!')
	print(resultMat_2)
	print(resultMat_2.shape)
	print('\n')


	resultDF=DataFrame(resultMat)         ## mat to DF 
	resultDF_2=DataFrame(resultMat_2)



	resultDF.to_csv(fileName+'_morning_predict_v1.csv')
	resultDF_2.to_csv(fileName+'_afternoon_predict_v1.csv')














'''

train = pd.read_csv("A2_72mat.csv")
test = pd.read_csv("A2_72mat01.csv")

train=train.drop('Unnamed: 0',axis=1)
test=test.drop('Unnamed: 0',axis=1)

train=np.array(train)    #DataFrame to ndarray
test=np.array(test)



train_feat=train[:,18:24]
train_id=train[:,24]

test_feat=test[:,18:24]
#test_id=test[24]

train_feat=np.mat(train_feat)   # ndarray to mat

train_id=np.mat(train_id).T    ## 列向量！！！！！！！！！！！！！！！！！！！！！！！！！


test_feat=np.mat(test_feat)   # ndarray to mat
#test_id=np.mat(test_id)




#print(train_feat.shape)#,train_id.shape,test_feat.shape)#,test_id.shape)

gbdt.fit(train_feat,train_id)


pred=gbdt.predict(test_feat)
'''
'''
print(pred)
total_err=0
'''
'''
for i in range(pred.shape[0]):
    print(pred[i],test_id[i])
    err=(pred[i]-test_id[i])/test_id[i]
    total_err+=err*err
'''

#print total_err/pred.shape[0]







