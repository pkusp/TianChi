#############################################

###  gbdt0520_v3.py 添加了weather数据，上午全部用09：00代替，下午全部用18：00代替
###  gbdt0521_v4.py 将时间窗口前移20min（train_x,train_y），使数据量翻倍
###
###  暂时放弃此脚本，本来准备手动划分验证集，感觉有点蠢，还是自动划分吧！！！！ 2017.05.21
##############################################
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

	###########################################################  morning    ！！！
	resultArr=[]
	#resultList=[]

	for i in range(6): 

		j = 24+i
		train_x01=train[:,18:24]   #时间特征
		train_x02=train[:,72:77]   #天气特征   #位于第72：76列

		train_x03=train[:,17:23]
		train_x04=train[:,72:77]   #天气特征   #位于第72：76列

		train_x05=train[:,19:25]
		train_x06=train[:,72:77]   #天气特征   #位于第72：76列	


		train_x_1 = np.hstack((train_x01,train_x02))  #合并特征数组 横向增加特征值
		train_x_2 = np.hstack((train_x03,train_x04))  #合并特征数组
		train_x_3 = np.hstack((train_x05,train_x06))  #合并特征数组


		train_y1=train[:,j]        #label 从第24列开始，共执行六次
		train_y2=train[:,j-1]
		train_y3=train[:,j+1]


		train_x=np.vstack((train_x_1,train_x_2,train_x_3)) #合并特征数组 纵向增加数据量
		train_y=np.hstack((train_y1,train_y2,train_y3))
		print(type(train_x))
		print(train_x.shape)
		print(type(train_y))

		### 划分验证集
		train_x=train_x[:220,:] ; valid_x=train_x[220:,:]  ##手动划分？？？？？ 嫌的无聊啊！！
		train_y=train_y[:220,:] ; valid_y=train_y[220:,:]  ##


		test_x01=test[:,18:24]     #测试数据特征    
		test_x02=test[:,72:77]    #天气特征       #位于第72：76列 
		test_x = np.hstack((test_x01,test_x02))  #合并特征数组
		#test_id=test[24]
		#print(test_x)
		#print('\n\n\n')

 		  
		train_x=np.mat(train_x) ; valid_x=np.mat(valid_x)           # ndarray to mat 
		train_y=np.mat(train_y).T ; valid_y=np.mat(train_y).T      ## 列向量！！！！！！！！！！！！！！！！！！！！！！！！！

		test_x=np.mat(test_x) ;   # ndarray to mat
		#test_id=np.mat(test_id)
		#print(train_feat.shape)#,train_id.shape,test_feat.shape)#,test_id.shape)

		gbdt.fit(train_x,train_y)
		pred=gbdt.predict(test_x)

		resultArr.append(pred)     #
	print('this is resultArr !!!!!!!!!!!!!!!!!!!!!!!!!!!')
	#print(resultArr)
	#resultMat=np.mat(resultArr)        # arr to mat
	#resultMat=resultMat.T           # .T
	resultArr = np.array(resultArr)
	resultList=list(resultArr[:,0])+list(resultArr[:,1])+list(resultArr[:,2])+list(resultArr[:,3])+list(resultArr[:,4])+list(resultArr[:,5])+list(resultArr[:,6])

	print('this result is:    '+fileName+'   new morning data prediction!!')
	#print(resultList)
	#print(resultList.shape)
	print('\n')

'''
	#######################################################   afternoon  ！！！
	resultArr_2=[]
	for i in range(6):        #######  15:00 to 17:00 predict 17:00 to 19:00  ##########

		j = 51+i
		
		#train_x01=train[:,45:51]   #特征 15:00 to 17:00
		#train_x02=train[:,77:82]   #天气特征
		#train_x = np.hstack((train_x01,train_x02))  #合并特征数组

		#train_y=train[:,j]        #label 从第51列开始，共执行六次
		
		train_x01=train[:,45:51]   #时间特征
		train_x02=train[:,77:82]   #天气特征   #位于第72：76列

		train_x03=train[:,44:50]
		train_x04=train[:,77:82]   #天气特征   #位于第72：76列

		train_x05=train[:,46:52]
		train_x06=train[:,77:82]   #天气特征   #位于第72：76列	


		train_x_1 = np.hstack((train_x01,train_x02))  #合并特征数组 横向增加特征值
		train_x_2 = np.hstack((train_x03,train_x04))  #合并特征数组
		train_x_3 = np.hstack((train_x05,train_x06))  #合并特征数组


		train_y1=train[:,j]        #label 从第24列开始，共执行六次
		train_y2=train[:,j-1]
		train_y3=train[:,j+1]


		train_x=np.vstack((train_x_1,train_x_2,train_x_3)) #合并特征数组 纵向增加数据量
		train_y=np.hstack((train_y1,train_y2,train_y3))





		test_x01=test[:,45:51]     #测试数据
		test_x02=test[:,77:82]    #天气特征
		test_x = np.hstack((test_x01,test_x02))  #合并特征数组
		#test_id=test[24]
		#print(test_x)
		#print('\n\n\n')



		train_x=np.mat(train_x)   # ndarray to mat
		test_y=np.mat(test_x)   # ndarray to mat
		#test_id=np.mat(test_id)
		train_y=np.mat(train_y).T    ## 列向量！！！！！！！！！！！！！！！！！！！！！！！！！

		print(train_x.shape,train_y.shape,test_x.shape)#,test_id.shape)

		gbdt.fit(train_x,train_y)
		pred=gbdt.predict(test_x)

		resultArr_2.append(pred)     #

	resultArr_2 = np.array(resultArr_2)
	resultList_2=list(resultArr_2[:,0])+list(resultArr_2[:,1])+list(resultArr_2[:,2])+list(resultArr_2[:,3])+list(resultArr_2[:,4])+list(resultArr_2[:,5])+list(resultArr_2[:,6])

	#resultMat_2=np.mat(resultArr_2)        # arr to mat
	#resultMat_2=resultMat_2.T           # .T

	print('this result is:   '+fileName+'   new afternoon data prediction!!')
	print(resultList_2)
	#print(resultList_2.shape)
	print('\n')




	###################################################### to csv


	resultDF=DataFrame(resultList)         ## mat to DF 
	resultDF_2=DataFrame(resultList_2)

	resultDF.to_csv(fileName+'_morning_predict_0521_v5.csv')
	resultDF_2.to_csv(fileName+'_afternoon_predict_0521_v5.csv')


'''











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







