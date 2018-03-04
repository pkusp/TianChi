#
#
#   model_gbdt 特征过于简单粗暴，拟合的结果太平稳了，显然有问题
#   2017.06.11 model_dnn  特征为power+weather+holiday，结果符合逻辑，但太不稳定
#
import numpy as np
import pandas as pd
#from sklearn.svm import SVR 
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from pandas import Series,DataFrame
#from sklearn.model_selection import train_test_split
#from sklearn import datasets
from sklearn import cross_validation
import tensorflow as tf 
import skflow
#import xgboost as xgb
#
# power_mat31_T:
#  0-------19
#  |
#  |
#  31
#
def run_gbdt(n):
	sum_score=0
	for i in range(n):
		sum_score+=gbdt_validation(train_x,train_y)
	return sum_score/n

def run_dnn(n):
	sum_score=0
	for i in range(n):
		sum_score+=dnn_validation(train_x,train_y)
	return sum_score/n






gbdt=GradientBoostingRegressor(             ####  上下午各一棵树
	loss='ls'                        #均方误差
	, learning_rate=0.0001
	, n_estimators=100
	, subsample=0.8
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

def value_fill(arr):
	for i in range(31):
		for j in range(21):
			if arr[i,j]==0:
				arr[i,j]=arr[i-1,j]  # 空缺值用前一天补齐

def data_pre(data):

	train=np.array(data)
	scar_1=train[:,:12]/100000
	scar_2=train[:,12:]

	train=np.hstack((scar_1,scar_2))
	test_x=train[279:,:16] # test data  test
	
	train_x=train[:279,:16]
	train_y=train[:279,16]/100000
	#print(train_x,train_y,test_x)
	return train_x,train_y,test_x

def gbdt_validation(x_name,label_name):                ### 验证函数
	x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
	gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！
	#pred=gbdt.predict(x_test)
	gbdt_score=gbdt.score(x_test,y_test)
	return gbdt_score

def gbdt_train(train_x,train_y,test_x):
		
	each_score=gbdt_validation(train_x,train_y)  
	print('\n\n the score:')
	print(each_score)

	gbdt.fit(train_x,train_y) 
	pred_gbdt=gbdt.predict(test_x)   ##  预测结果，用test表示，
	##############################
	return pred_gbdt

def dnn_validation(train_x,trian_y):  ## return MSE
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=16)]
	dnn_cv = skflow.DNNRegressor(feature_columns=feature_columns,hidden_units=[20,20,20,20],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),activation_fn=tf.nn.relu)
	
	train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
	
	dnn_cv.fit(x=train_X_cv,y=train_Y_cv,batch_size=300,steps=100000)
	pred = dnn_cv.predict(x=test_X_cv)
	#print('\n\n')
	print('\n\n')
	#print(pred)
	pred=list(pred)
	mse=((test_Y_cv-pred)**2).mean()
	return mse




def dnn(train_x,train_y,test_x):

			feature_columns = [tf.contrib.layers.real_valued_column("", dimension=16)]
			tf_dnn_regressor=skflow.DNNRegressor(feature_columns=feature_columns,hidden_units=[16,20,50,50,20,10],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),activation_fn=tf.nn.relu)
			##################  CV   ##########################
			#dnn_cv = skflow.DNNRegressor(feature_columns=feature_columns,hidden_units=[10,10,10,10],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),activation_fn=tf.nn.relu)
			
			#train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
			
			#dnn_cv.fit(x=train_X_cv,y=train_Y_cv,batch_size=50,steps=1000)
			#pred = dnn_cv.predict(x=test_X_cv)
			#print('\n\n')
			print('\n\n')
			#print(pred)
			#pred=list(pred)
			#mse=((test_Y_cv-pred)**2).mean()

			#print(pred)
			###################  CV   ###########################

			tf_dnn_regressor.fit(x=train_x,y=train_y,batch_size=300,steps=100000)
			pred_dnn = tf_dnn_regressor.predict(x=test_x)
			pred_dnn = list(pred_dnn)
			pred_dnn = np.array(pred_dnn)
			###########################################################################
			#resultArr.append(pred_rfr)     ##   导出结果
			return pred_dnn#,mse




data = pd.read_csv('train_xy_mat.csv')
data = data.drop('Unnamed: 0',axis=1)
print(data)
train_x,train_y,test_x=data_pre(data)


#pred_label=gbdt_train(train_x,train_y,test_x)

pred_label_1=dnn(train_x,train_y,test_x)
pred_label_2=dnn(train_x,train_y,test_x)
pred_label_3=dnn(train_x,train_y,test_x)
pred_label_4=dnn(train_x,train_y,test_x)
pred_label_5=dnn(train_x,train_y,test_x)

pred_label=(pred_label_5+pred_label_4+pred_label_3+pred_label_2+pred_label_1)/5


pred_result=pred_label*100000#[:-1]

print('the result:')
print(pred_result)
#print('the mse is',mse/100000000)

pred_result=DataFrame(pred_result)
pred_result.to_csv('result.csv')


#print(pred_label)

#table=pd.read_csv('Tianchi_power_predict_table.csv')
#table['predict_power_consumption']=pred_result

#table.to_csv('Tianchi_power_predict_table.csv',index=None) ## index=none 防止多导出一列




