#############################################

###  gbdt0520_v3.py 添加了weather数据，上午全部用09：00代替，下午全部用18：00代替
###  gbdt0521_v4.py 将时间窗口前移20min（train_x,train_y），使数据量翻倍
###
###  gbdt0521_v6.py 增加cross_validation，n次运行cross_validation保证误差稳定、
###  将上下午分开建树，采取不同的参数
###  增加MAPE函数（单条路）  
###  0523_v6.csv 修改n_estimators：100 to 20
##############################################
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from pandas import Series,DataFrame
#from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import cross_validation


def MAPE(act,prd):
	err=abs(((act-prd)/act)).mean()
	return err
'''
def gbdt_validation(x_name,label_name):                ### 验证函数
	x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
	gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！
			
	pred1=gbdt.predict(x_test)
	print('          ########################      M    A     P   E  #############')
	print(MAPE(y_test,pred1))
			
	gbdt_score=gbdt.score(x_test,y_test)
	return gbdt_score
'''


def runNtimes(fileName,n):   ## 多次run err取平均，使err稳定
	avgScore_1=0
	avgScore_2=0
	for i in range(n):
		sumScore_1=morning_gbdt(fileName)   ### 调用each_gbdt 
		avgScore_1+=sumScore_1
	for i in range(n):
		sumScore_2=afternoon_gbdt(fileName)   ### 调用each_gbdt 
		avgScore_2+=sumScore_2
	avgScore_1=avgScore_1/n
	avgScore_2=avgScore_2/n
	return avgScore_1,avgScore_2

def morning_gbdt(fileName):     #return sumScore
	
	gbdt=GradientBoostingRegressor(             ####  上下午各一棵树
	loss='ls'                        #均方误差
	, learning_rate=0.05
	, n_estimators=20
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

	#######################################################   load data ！
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
	sumScore=0

	for i in range(6): 				### 六个时间窗口分别训练并验证

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

		test_x01=test[:,18:24]     #测试数据特征    
		test_x02=test[:,72:77]    #天气特征       #位于第72：76列 
		test_x = np.hstack((test_x01,test_x02))  #合并特征数组
		#test_id=test[24]
		#print(test_x)
		#print('\n\n\n')
		train_x=np.mat(train_x)   # ndarray to mat
		test_x=np.mat(test_x)   # ndarray to mat      #预测结果，用test表示
		#test_id=np.mat(test_id)

		#train_y=np.mat(train_y).T    ## 列向量！！！！！！！！！  ## 竟然不要列向量！！
		#print(train_feat.shape)#,train_id.shape,test_feat.shape)#,test_id.shape)

		#x_train, x_test, y_train, y_test = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation
		#gbdt.fit(x_train,y_train)  ####  GBDT训练在这开始 ！！
		#gbdt_score=gbdt.score(x_test,y_test)
		
		def gbdt_validation(x_name,label_name):                ### 验证函数
			x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
			gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！
			
			pred1=gbdt.predict(x_test)
			print('          ########################      M    A     P   E  #############')
			print(MAPE(y_test,pred1))
			
			gbdt_score=gbdt.score(x_test,y_test)
			return gbdt_score
		
		each_score=gbdt_validation(train_x,train_y)  ### 每个时间窗口的误差
		sumScore+=abs(each_score)                    ### sumScore为全局变量

		#print('\n')
		print(fileName + '第 '+ str(float(i)) + '个时间窗口的 平方误差 为：：')
		print(each_score)
		#print('\n')
		gbdt.fit(train_x,train_y) 
		pred=gbdt.predict(test_x)   ##  预测结果，用test表示，
		resultArr.append(pred)     ##   预测结果

	#print('\n')
	print(fileName+'的morning总误差为：')
	print(sumScore)
	#resultMat=np.mat(resultArr)        # arr to mat
	#resultMat=resultMat.T           # .T
	resultArr = np.array(resultArr)  ###### 为了方便导出csv
	resultList=list(resultArr[:,0])+list(resultArr[:,1])+list(resultArr[:,2])+list(resultArr[:,3])+list(resultArr[:,4])+list(resultArr[:,5])+list(resultArr[:,6])
	
	#print('\n')
	print('this result is: '+fileName+'new morning data prediction!!')
	#print('\n')

	###################################################### to csv
	resultDF=DataFrame(resultList)         ## mat to DF 
	resultDF.to_csv(fileName+'_morning_predict_0523_v1.csv')
	return sumScore

def afternoon_gbdt(fileName):

	gbdt=GradientBoostingRegressor(             ####  上下午各一棵树
	loss='ls'                        #均方误差
	, learning_rate=0.1
	, n_estimators=20
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


	train_name = fileName+'_72mat.csv'
	test_name = fileName + '_72mat01.csv'
	train = pd.read_csv(train_name)
	test = pd.read_csv(test_name)

	train=train.drop('Unnamed: 0',axis=1)   ### 读出的csv系统会自动添加索引，故drop it
	test=test.drop('Unnamed: 0',axis=1)

	train=np.array(train)    #DataFrame to ndarray
	test=np.array(test)

	###########################################################  afternoon    ！！！
	resultArr=[]
	#resultList=[]
	sumScore=0
	for i in range(6): 				### 六个时间窗口分别训练并验证
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
		test_x =np.mat(test_x)

		#train_y=np.mat(train_y).T    ## 列向量！！！！！！！！！  ## 竟然不要列向量！！
		#print(train_feat.shape)#,train_id.shape,test_feat.shape)#,test_id.shape)

		#x_train, x_test, y_train, y_test = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation
		#gbdt.fit(x_train,y_train)  ####  GBDT训练在这开始 ！！
		#gbdt_score=gbdt.score(x_test,y_test)
		def gbdt_validation(x_name,label_name):                ### 验证函数
			x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
			gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！
			pred2=gbdt.predict(x_test)

			print('          ########################      M    A     P   E  #############')
			print(MAPE(y_test,pred2))
			
			gbdt_score=gbdt.score(x_test,y_test)
			return gbdt_score

		each_score=gbdt_validation(train_x,train_y)  ### 每个时间窗口的误差
		sumScore+=abs(each_score)                    ### sumScore为全局变量

		#print('\n')
		print(fileName + '第 '+ str(float(i)) + '个时间窗口的 平方误差 为：：')
		print(each_score)
		#print('\n')

		gbdt.fit(train_x,train_y) 
		pred=gbdt.predict(test_x)   ##  预测结果，用test表示， test_x 在此处用到 ！！！
		resultArr.append(pred)     ##   预测结果

	#print('\n')
	print(fileName+'的afternoon总误差为：')
	print(sumScore)
	#print('this is resultArr !!!!!!!!!!!!!!!!!!!!!!!!!!!')
	#print(resultArr)
	#resultMat=np.mat(resultArr)        # arr to mat
	#resultMat=resultMat.T           # .T
	resultArr = np.array(resultArr)
	resultList_2=list(resultArr[:,0])+list(resultArr[:,1])+list(resultArr[:,2])+list(resultArr[:,3])+list(resultArr[:,4])+list(resultArr[:,5])+list(resultArr[:,6])
	
	#print('\n')
	print('this result is:    '+fileName+'   new afternoon data prediction!!')
	#print(resultList)
	#print(resultList.shape)
	#print('\n')

	resultDF_2=DataFrame(resultList_2)
	resultDF_2.to_csv(fileName+'_afternoon_predict_0523_v1.csv')

	return sumScore
























'''
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










