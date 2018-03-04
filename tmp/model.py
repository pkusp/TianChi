#############################################
#
###  
##############################################
import pandas as pd
#from sklearn.svm import SVR 
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
import numpy as np
#from sklearn.ensemble import GradientBoostingRegressor
from pandas import Series,DataFrame
#from sklearn.model_selection import train_test_split
#from sklearn import datasets
#from sklearn import cross_validation
#import tensorflow as tf 
#import skflow
import xgboost as xgb
from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
######################################################  数据标准化  
#ss_x=StandardScaler()
#ss_y=StandardScaler()
import dataPre

def MSE(yact,yhat):
	return ((yhat-yact)**2).mean()

def loadData(fileName):
	data=pd.read_csv(fileName)
	return data

def xgb_validation(x_name,label_name):   	
	train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
	xg_train = xgb.DMatrix( train_X_cv, label=train_Y_cv)
	xg_test = xgb.DMatrix(test_X_cv)#, label=test_Y_cv)
	#watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
	#num_round = 500
	bst = xgb.train(param, xg_train, num_round)#, watchlist );   ### TRAIN
	# get prediction
	pred = bst.predict( xg_test )
	xgb_cv_score=bst.cv(test_X_cv,test_Y_cv)  ####  此处应该有
	return xgb_cv_score

def xgb_model(train_x,train_y,test_x,valid_x):

	param={
		'booster':'gbtree',
		'objective': 'reg:linear', #多分类的问题
		#'num_class':10, # 类别数，与 multisoftmax 并用
		'n_estimators':100,
		#'reg_alpha':1,
		#'reg_lambda':200,

		'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
		'max_depth':5, # 构建树的深度，越大越容易过拟合
		'lambda':200,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
		'alpha':10,
		'subsample':0.9, # 随机采样训练样本
		'colsample_bytree':0.9, # 生成树时进行的列采样
		'min_child_weight':1, 
		# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
		#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
		#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
		'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
		'eta': 0.01, # 如同学习率
		'seed':500,
		#'nthread':7,# cpu 线程数
		#'eval_metric': 'auc'
		}

	#print(i,' .. ')
	num_round=50000
	#train_final=xgb.DMatrix(train_x,label=train_y)   ## 数据格式转换
	#xgb_cv_score_02=xgb.cv(param, xg_train, num_round, nfold=5,metrics={'error'}, seed = 0,callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
	#xgbloss=xgb_validation(train_x,train_y)
	print('\n')
	#print('xgb loss: ', xgbloss)
	print('\n')
	#xgb_cv_score_02=xgb.cv(param, train_final, num_round, nfold=5,metrics={'error'}, seed = 0,callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
	test_final =xgb.DMatrix(test_x)
	train_final=xgb.DMatrix(train_x,label=train_y)   ## 数据格式转换
	watchlist=[(train_final,'train')]
	xgb_final=xgb.train(param,train_final,num_round,watchlist)

	pred_xgb=xgb_final.predict( test_final )

	valid_x=xgb.DMatrix(valid_x)
	pred_val=xgb_final.predict(valid_x)
	mse=MSE(valid_y,pred_val)
	print('mse:',mse)
	return pred_xgb #,yhat_xgb

train_test_DF=loadData('afterPCA_full_data.csv')
train_test_arr=np.array(train_test_DF)
train_x=train_test_arr[:500,1:41]
train_y=train_test_arr[:500,41:42]

valid_x=train_test_arr[400:500,1:41]
valid_y=train_test_arr[400:500,41:42]

test_x=train_test_arr[500:,1:41]
print('x:',train_x.shape)
print('y:',train_y.shape)
print('test:',test_x.shape)

result=xgb_model(train_x,train_y,test_x,valid_x)
print(result)

resultDF=DataFrame(result)
resultDF.to_csv('result_20171220.csv')




def dataP(fileName):
	def loadData(fileName):




		data=pd.read_csv(fileName)
		return data

	dataFrame=loadData('train.csv')  # 500*8029
	dataArr=np.array(dataFrame)
	dataMatrix=np.mat(dataFrame)

	testFrame=loadData('testA.csv')
	testArr=np.array(testFrame)

	dataArr0=dataArr[:,2:] # 去除第零列ID号和第一列tool ID  (500*8027)
	testArr0=testArr[:,2:]

	train_x=np.mat(dataArr0[:,8026])	# 500*8026
	train_y=np.mat(dataArr0[:,8026:8027])	# 500*1
	test_x=np.mat(testArr0)				# 100*8026

	print(test_x)



'''

def morning_stacking(fileName):     #return sumScore
		
	#######################################################   load data ！
	train_name = fileName+'_72matTrain.csv'
	test_name = fileName + '_swapTest__72mat.csv'
	train = pd.read_csv(train_name)                  ####### train data
	test = pd.read_csv(test_name)                    ####### test data

	train=train.drop('Unnamed: 0',axis=1)
	test=test.drop('Unnamed: 0',axis=1)

	train=np.array(train)    #DataFrame to ndarray
	test=np.array(test)

	###########################################################  morning    ！！！
	resultArr=[]
	sumScore=0
	DNNmapeValue=0
	for i in range(6): 				### 六个时间窗口分别训练并验证
		#####################################  pre data   ##################
		def data_pre_AM():
	
			j = 24+i

			train_old=train   ### 保存train数组，后续test数组需使用原数组10.05到10.17号的数据
			###############################################################################   TRAIN data 处理   ########################
			train_day_07=train[6:89,:72]    #七天之前的数组     ###### 以下分别求一周前的数组
			train_day_08=train[5:88,:72]    #8天之前的数组
			train_day_09=train[4:87,:72]    #9天之前的数组
			train_day_10=train[3:86,:72]    #10天之前的数组
			train_day_11=train[2:85,:72]    #11天之前的数组
			train_day_12=train[1:84,:72]    #12天之前的数组
			train_day_13=train[:83,:72]     #13天之前的数组

			train_76=train[14:,:72]          #############  train 切割为 76 行
			train_weather=train_old[14:,72:77]     #选中weather,要和train_76维度一样  ！！
			train_holiday=train_old[14:,82:83]     ## holiday
			train_weather=ss_x.fit_transform(train_weather)

			################################################  以下合并为大数组，包含八天数据 ######################
			train_time=np.hstack((train_76,train_day_07,train_day_08,train_day_09,train_day_10,train_day_11,train_day_12,train_day_13,))
			train_all_data=np.hstack((train_time,train_weather,train_holiday))   #大数组和weather整合,weather在最后   
			###################################################################################################################
			x00_train=train_all_data[:,18:24]       #当天6到8点          ####   以下提取特征 time（6+1*7）+weather（5）  ########
			xday07_train=train_all_data[:,(j+72):(j+72+1)]     #七天前本时间段 一列
			xday08_train=train_all_data[:,(j+72*2):(j+72*2+1)]   #八天前        一列!!!
			xday09_train=train_all_data[:,(j+72*3):(j+72*3+1)]	#九天
			xday10_train=train_all_data[:,(j+72*4):(j+72*4+1)]	#十天
			xday11_train=train_all_data[:,(j+72*5):(j+72*5+1)]	#十一天
			xday12_train=train_all_data[:,(j+72*6):(j+72*6+1)]	#十二天
			xday13_train=train_all_data[:,(j+72*7):(j+72*7+1)]	#十三天
			##################################################  以下特征合并为数组   ########################
			train_x_final=np.hstack((x00_train,xday07_train,xday08_train,xday09_train,xday10_train,xday11_train,xday12_train,xday13_train,train_weather,train_holiday))
			train_y_final=train_76[:,j]   #trian[]已经降为76行
			####################################################################
			train_x=np.mat(train_x_final)   ##转化为原名，方便后续函数调用
			train_y=train_y_final
			###############################################################################   TRAIN data 处理结束   ########################


			################################################################################   TEST data 处理  #########################
			#test_old=test[:,:72]                  ######### 以下分别求 一周前数组 test_old 为原test数组
			test_day07=train_old[90:,:72]   		##利用train[]内的数据  一周
			test_day08=train_old[89:96,:72]
			test_day09=train_old[88:95,:72]
			test_day10=train_old[87:94,:72]
			test_day11=train_old[86:93,:72]
			test_day12=train_old[85:92,:72]
			test_day13=train_old[84:91,:72]

			test_old=test[:,:72]  
			test_weather=test[:,72:77]
			test_holiday=test[:,82:83]
			test_weather=ss_x.transform(test_weather)

			################################################### 以下合并为大数组，包含8天数据
			test_time=np.hstack((test_old,test_day07,test_day08,test_day09,test_day10,test_day11,test_day12,test_day13,))
			test_all_data=np.hstack((test_time,test_weather,test_holiday))

			###################################################################################################################
			x00_test=test_all_data[:,18:24]       #当天6到8点          ####   以下提取特征 time（6+1*7）+weather（5）  ########
			xday07_test=test_all_data[:,j+72:j+72+1]     #七天前本时间段 一列
			xday08_test=test_all_data[:,j+72*2:j+72*2+1]   #八天前        一列
			xday09_test=test_all_data[:,j+72*3:j+72*3+1]	#九天
			xday10_test=test_all_data[:,j+72*4:j+72*4+1]	#十天
			xday11_test=test_all_data[:,j+72*5:j+72*5+1]	#十一天
			xday12_test=test_all_data[:,j+72*6:j+72*6+1]	#十二天
			xday13_test=test_all_data[:,j+72*7:j+72*7+1]	#十三天
			##################################################  以下特征合并为数组   ########################
			test_x_final=np.hstack((x00_test,xday07_test,xday08_test,xday09_test,xday10_test,xday11_test,xday12_test,xday13_test,test_weather,test_holiday))
			test_y_final=test_old[:,j]   #trian[]已经降为76行

			test_x=np.mat(test_x_final)
			################################################################################   TEST data 处理结束   ##########################
			return train_x,train_y,test_x
		########################################################  stacking  ###############	
		train_x,train_y,test_x = data_pre_AM()

		def xgb_01(train_x,train_y,test_x):
	
			param={
				'booster':'gbtree',
				'objective': 'reg:linear', #多分类的问题
				#'num_class':10, # 类别数，与 multisoftmax 并用
				'n_estimators':100,
				#'reg_alpha':1,
				#'reg_lambda':200,

				'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
				'max_depth':6, # 构建树的深度，越大越容易过拟合
				'lambda':200,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
				'alpha':10,
				'subsample':0.7, # 随机采样训练样本
				'colsample_bytree':0.7, # 生成树时进行的列采样
				'min_child_weight':1, 
				# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
				#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
				#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
				'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
				'eta': 0.2, # 如同学习率
				'seed':500,
				#'nthread':7,# cpu 线程数
				#'eval_metric': 'auc'
				}

			print(i,' .. ')
			num_round=500
			#train_final=xgb.DMatrix(train_x,label=train_y)   ## 数据格式转换

		
			def xgb_validation(x_name,label_name):   
			
				train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			

				xg_train = xgb.DMatrix( train_X_cv, label=train_Y_cv)
				xg_test = xgb.DMatrix(test_X_cv)#, label=test_Y_cv)
				#watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
				#num_round = 500
				bst = xgb.train(param, xg_train, num_round)#, watchlist );   ### TRAIN
				# get prediction
				pred = bst.predict( xg_test )
				each_mape=MAPE(test_Y_cv,pred)
				print('each_mape')
				#xgb_cv_score=bst.cv(test_X_cv,test_Y_cv)  ####  此处应该有
				return each_mape

			#xgb_cv_score_02=xgb.cv(param, xg_train, num_round, nfold=5,metrics={'error'}, seed = 0,callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
			xgbMape=xgb_validation(train_x,train_y)
		
			#xgb_cv_score_02=xgb.cv(param, train_final, num_round, nfold=5,metrics={'error'}, seed = 0,callbacks=[xgb.callback.print_evaluation(show_stdv=False)])
			test_final =xgb.DMatrix(test_x)
			#sumScore+=xgb_cv_score_02
			train_final=xgb.DMatrix(train_x,label=train_y)   ## 数据格式转换
			#watchlist=[(train_final,'train')]
			xgb_final=xgb.train(param,train_final,num_round)#,watchlist)

			pred_xgb=xgb_final.predict( test_final )
			train_x=xgb.DMatrix(train_x)
			yhat_xgb=xgb_final.predict( train_x )
			#########################################################  xgb over #################
			#resultArr.append(pred_xgb)     ##   预测结果
			return pred_xgb,yhat_xgb

		def gbdt_02(train_x,train_y,test_x):
	
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

			def gbdt_validation(x_name,label_name):                ### 验证函数
				x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
				gbdt.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！

				pred=gbdt.predict(x_test)
				each_mape=MAPE(y_test,pred)
				print('\n\n\n\n')
				print(each_mape)
				print('\n\n\n\n')
				#print('          ########################      M    A     P   E  #############')			mapeErr=MAPE(y_test,pred1)
				gbdt_score=gbdt.score(x_test,y_test)
				return gbdt_score#,mapeErr
			
			each_score=gbdt_validation(train_x,train_y)  ### 每个时间窗口的误差
			#sumScore+=abs(each_score)                    ### sumScore为全局变量
			#mapeValue+=each_mape
			#print('\n')
			gbdt.fit(train_x,train_y) 

			pred_gbdt=gbdt.predict(test_x)   ##  预测结果，用test表示，
			yhat_gbdt=gbdt.predict(train_x)
			############################################################################
			#resultArr.append(pred_gbdt)     ##   预测结果
			return pred_gbdt,yhat_gbdt

		def svr_03(train_x,train_y,test_x):
			rbf_svr=SVR(kernel='rbf')
			##################  CV   ##########################
			rbf_cv=SVR(kernel='rbf')
			train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
			rbf_cv.fit(train_X_cv,train_Y_cv)
			pred=rbf_cv.predict(test_X_cv)
			each_mape=MAPE(test_Y_cv,pred)    # cv 验证MAPE
			print(each_mape)
			##################  CV   ##########################
			rbf_svr.fit(train_x,train_y)
			pred_svr=rbf_svr.predict(test_x)
			yhat_svr=rbf_svr.predict(train_x)   ### yhat
			###########################################################################
			#resultArr.append(pred_svr)     ##   导出结果
			return pred_svr,yhat_svr
		def knr_04(train_x,train_y,test_x):
			dis_knr=KNeighborsRegressor(weights='distance')
			##################  CV   ##########################
			knr_cv=KNeighborsRegressor(weights='distance')
			train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
			knr_cv.fit(train_X_cv,train_Y_cv)
			pred=knr_cv.predict(test_X_cv)
			each_mape=MAPE(test_Y_cv,pred)    # cv 验证MAPE
			print(each_mape)
			###################  CV   ###########################
			dis_knr.fit(train_x,train_y)
			pred_knr=dis_knr.predict(test_x)
			yhat_knr=dis_knr.predict(train_x)  ### yhat
			###########################################################################
			#resultArr.append(pred_knr)     ##   导出结果
			return pred_knr,yhat_knr
		def linear_05(train_x,train_y,test_x):
			lr=LinearRegression()
			##################  CV   ##########################
			lr_cv=LinearRegression()
			train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
			lr_cv.fit(train_X_cv,train_Y_cv)
			pred=lr_cv.predict(test_X_cv)
			each_mape=MAPE(test_Y_cv,pred)    # cv 验证MAPE
			print(each_mape)
			###################  CV   ###########################
			lr.fit(train_x,train_y)
			pred_lr=lr.predict(test_x)
			yhat_lr=lr.predict(train_x)  ### yhat
			###########################################################################
			#resultArr.append(pred_lr)     ##   导出结果
			return pred_lr,yhat_lr
		def rfr_06(train_x,train_y,test_x):
			rfr=RandomForestRegressor()
			##################  CV   ##########################
			rfr_cv=RandomForestRegressor()
			train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
			rfr_cv.fit(train_X_cv,train_Y_cv)
			pred=rfr_cv.predict(test_X_cv)
			each_mape=MAPE(test_Y_cv,pred)    # cv 验证MAPE
			print(each_mape)
			###################  CV   ###########################
			rfr.fit(train_x,train_y)
			pred_rfr=rfr.predict(test_x)
			yhat_rfr=rfr.predict(train_x)  ### yhat
			###########################################################################
			#resultArr.append(pred_rfr)     ##   导出结果
			return pred_rfr,yhat_rfr
		def dnn_07(train_x,train_y,test_x):

			feature_columns = [tf.contrib.layers.real_valued_column("", dimension=18)]
			tf_dnn_regressor=skflow.DNNRegressor(feature_columns=feature_columns,hidden_units=[5,5,5],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),activation_fn=tf.nn.relu)
			##################  CV   ##########################
			dnn_cv = skflow.DNNRegressor(feature_columns=feature_columns,hidden_units=[5,5,5],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),activation_fn=tf.nn.relu)
			train_X_cv, test_X_cv, train_Y_cv, test_Y_cv = cross_validation.train_test_split(train_x, train_y, test_size=0.2, random_state=0)  # validation			
			dnn_cv.fit(x=train_X_cv,y=train_Y_cv,batch_size=10,steps=1000)
			pred = dnn_cv.predict(x=test_X_cv)
			print('\n\n')
			print('\n\n')
			print(pred)
			pred=list(pred)
			print(pred)
			print('\n\n\n\n')
			each_mape = MAPE(test_Y_cv,pred)    # cv 验证MAPE
			print(each_mape)
			###################  CV   ###########################

			tf_dnn_regressor.fit(x=train_x,y=train_y,batch_size=10,steps=1000)
			pred_dnn = tf_dnn_regressor.predict(x=test_x)
			pred_dnn = list(pred_dnn)
			pred_dnn = np.array(pred_dnn)
			yhat_dnn=tf_dnn_regressor.predict(train_x)  ### yhat
			yhat_dnn=list(yhat_dnn)
			yhat_dnn=np.array(yhat_dnn)
			###########################################################################
			#resultArr.append(pred_rfr)     ##   导出结果
			return pred_dnn,yhat_dnn

		#########################################################  stacking over #################
		print('\n')

		pred1,yhat1=xgb_01(train_x,train_y,test_x)
		pred2,yhat2=gbdt_02(train_x,train_y,test_x)
		pred3,yhat3=svr_03(train_x,train_y,test_x)
		pred4,yhat4=knr_04(train_x,train_y,test_x)
		pred5,yhat5=linear_05(train_x,train_y,test_x)
		pred6,yhat6=rfr_06(train_x,train_y,test_x)
		pred7,yhat7=dnn_07(train_x,train_y,test_x)
		#############################################################################  stacking train start !!!!

		yhat1=np.array(yhat1);pred1=np.array(pred1)  ###  to ndarray
		yhat2=np.array(yhat2);pred2=np.array(pred2)
		yhat3=np.array(yhat3);pred3=np.array(pred3)
		yhat4=np.array(yhat4);pred4=np.array(pred4)
		yhat5=np.array(yhat5);pred5=np.array(pred5)
		yhat6=np.array(yhat6);pred6=np.array(pred6)
		yhat7=np.array(yhat7);pred7=np.array(pred7)

		yhat1=np.mat(yhat1).T; pred1=np.mat(pred1).T
		yhat2=np.mat(yhat2).T; pred2=np.mat(pred2).T
		yhat3=np.mat(yhat3).T; pred3=np.mat(pred3).T
		yhat4=np.mat(yhat4).T; pred4=np.mat(pred4).T
		yhat5=np.mat(yhat5).T; pred5=np.mat(pred5).T
		yhat6=np.mat(yhat6).T; pred6=np.mat(pred6).T
		yhat7=np.mat(yhat7).T; pred7=np.mat(pred7).T


		train_x,train_y,test_x = data_pre_AM()

		stacking_train = np.hstack((yhat1,yhat2,yhat3,yhat4,yhat5,yhat6,yhat7))  ###  data
		stacking_label = train_y
		stacking_test = np.hstack((pred1,pred2,pred3,pred4,pred5,pred6,pred7))

		stacking_train = np.hstack((train_x,stacking_train))  ##  合并特征，用train_x 和 yhat array共同做特征
		stacking_test = np.hstack((test_x,stacking_test))

		###----------------------------------------------------------------
		#def stacking_train_model(stacking_train,stacking_label,stacking_test):
		def gbdt_stacking(train_x,train_y,test_x):
	
			gbdt_stacking=GradientBoostingRegressor(             ####  上下午各一棵树
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

			def gbdt_validation(x_name,label_name):                ### 验证函数
				x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_name, label_name, test_size=0.2, random_state=0)  # validation			
				gbdt_stacking.fit(x_train,y_train)                          ####  GBDT训练在这开始 ！！

				pred=gbdt_stacking.predict(x_test)
				each_mape=MAPE(y_test,pred)
				print('\n\n stacking is here !!!\n\n')
				print(each_mape)
				print('\n\n\n\n')
				#print('          ########################      M    A     P   E  #############')			mapeErr=MAPE(y_test,pred1)
				return each_mape#gbdt_score#,mapeErr
			
			each_mape=gbdt_validation(train_x,train_y)  ### 每个时间窗口的误差
			#sumScore+=abs(each_score)                    ### sumScore为全局变量
			#mapeValue+=each_mape
			#print('\n')
			gbdt_stacking.fit(train_x,train_y) 

			pred_gbdt=gbdt_stacking.predict(test_x)   ##  预测结果，用test表示，
			#yhat_gbdt=gbdt_stacking.predict(train_x)
			############################################################################
			#resultArr.append(pred_gbdt)     ##   预测结果
			return pred_gbdt#,yhat_gbdt

		pred_stacking=gbdt_stacking( stacking_train,stacking_label,stacking_test )  ## model return predict


		#####################################  stacking  end !!!!
		resultArr.append(pred_stacking)

		#pred_final= 0.1*pred1+0.1*pred2 + 0.4*pred3+0.1*pred4+0.1*pred5+0.1*pred6+0.1*pred7
		#resultArr.append(pred_final)     ##   stacking结果
		#########################################################  stacking end  #################

		#####################################################  real stacking here !!!  #####################
	#resultMat=resultMat.T           # .T
	resultArr = np.array(resultArr)  ###### 为了方便导出csv
	resultList=list(resultArr[:,0])+list(resultArr[:,1])+list(resultArr[:,2])+list(resultArr[:,3])+list(resultArr[:,4])+list(resultArr[:,5])+list(resultArr[:,6])

	###################################################### to csv
	resultDF=DataFrame(resultList)         ## mat to DF 
	resultDF.to_csv(fileName+'_AM_predict_stacking_yhat+x+holiday+ss.csv')
	return 0#xgbMape#,sumScore
'''


