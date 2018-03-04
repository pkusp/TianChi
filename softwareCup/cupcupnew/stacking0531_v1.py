#############################################
###  gbdt0520_v3.py 添加了weather数据，上午全部用09：00代替，下午全部用18：00代替
###  gbdt0521_v4.py 将时间窗口前移20min（train_x,train_y），使数据量翻倍
###
###  gbdt0521_v6.py 增加cross_validation，n次运行cross_validation保证误差稳定、
###  将上下午分开建树，采取不同的参数
###  增加MAPE函数（单条路）  
###  0523_v6.csv 修改n_estimators：100 to 20
###
###  0523_v2.py  增加特征 前一周同时间段（1*7），增加MAPE测评，n_estimators 为 6 时效果较好
###
###  gbdt0528.py  为kddfinal使用的代码，添加新数据，尝试修改loss functi in params of gbdt
###
###  xgb0529改自gbdt0528
###
###  stacking0530 引入五个model，加权平均，分别为 SVR, KNR, LR, RFR, DNN 
###  
###  next
###  下一步重新加入 GBR 和 XGBoost
###  有时间加入ARIMA
###  8个model做真正的stacking
###  
##############################################
import pandas as pd
from sklearn.svm import SVR 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from pandas import Series,DataFrame
#from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import cross_validation
import tensorflow as tf 
import skflow
import xgboost as xgb


def runNtimes(fileName,n):   ## 多次run err取平均，使err稳定
	avgScore_1=0
	avgScore_2=0
	avgMape_1 =0
	avgMape_2 =0
	for i in range(n):
		mapeValue_1=morning_stacking(fileName)   ### 调用each_gbdt 
		#avgScore_1+=sumScore_1
		avgMape_1+=mapeValue_1
	for i in range(n):
		mapeValue_2=afternoon_stacking(fileName)   ### 调用each_gbdt 
		#avgScore_2+=sumScore_2
		avgMape_2+=mapeValue_2
	#avgScore_1=avgScore_1/n
	#avgScore_2=avgScore_2/n
	avgMape_1 =avgMape_1/n
	avgMape_2 =avgMape_2/n
	return avgMape_1,avgMape_2


def MAE(act,prd):
	err=abs(act-prd).mean()
	return err

def MAPE(act,prd):
	act=act+1
	err=abs((act-prd)/act).mean()
	return err

def data_pre(fileName):
	data_sale=pd.read_csv(fileName)
	#data_rank=pd.read_csv('fileName2')	#格式有问题
	data_sales=data_sale.drop('Unnamed: 0',axis=1)

	train_arr=np.array(data_sales) # train_arr为原始91列数据
	train_v=train_arr[:,1:51] #	取arr第一块作为初始训练集

	for i in range(2,31):		# 将所有训练集vstack
		train_each=train_arr[:,i:i+50]
		train_v=np.vstack((train_v,train_each))

	train_x=train_v[:,:49]
	train_y=train_v[:,49:50]
	print('data pre end...')
	return train_x,train_y,train_arr


train_x,train_y,train_arr=data_pre('round_mat_T_sales.csv')



def model_train(fileName):     #return sumScore
		
	#######################################################   load data ！

	###########################################################  morning    ！！！
	resultArr=[]
	sumScore=0
	DNNmapeValue=0

		########################################################  stacking  ###############	
		train_x,train_y,test_x 

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
			tf_dnn_regressor=skflow.DNNRegressor(feature_columns=feature_columns,hidden_units=[20,20,20],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),activation_fn=tf.nn.relu)
			##################  CV   ##########################
			dnn_cv = skflow.DNNRegressor(feature_columns=feature_columns,hidden_units=[20,20,20],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),activation_fn=tf.nn.relu)
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
		pred6,yhat6=rfr_06(train_x,train_y,test_x)
		pred7,yhat7=dnn_07(train_x,train_y,test_x)
		#############################################################################  stacking train start !!!!

		yhat1=np.array(yhat1);pred1=np.array(pred1)  ###  to ndarray
		yhat2=np.array(yhat2);pred2=np.array(pred2)
		yhat3=np.array(yhat3);pred3=np.array(pred3)
		yhat4=np.array(yhat4);pred4=np.array(pred4)
		yhat6=np.array(yhat6);pred6=np.array(pred6)
		yhat7=np.array(yhat7);pred7=np.array(pred7)

		yhat1=np.mat(yhat1).T; pred1=np.mat(pred1).T
		yhat2=np.mat(yhat2).T; pred2=np.mat(pred2).T
		yhat3=np.mat(yhat3).T; pred3=np.mat(pred3).T
		yhat4=np.mat(yhat4).T; pred4=np.mat(pred4).T
		yhat6=np.mat(yhat6).T; pred6=np.mat(pred6).T
		yhat7=np.mat(yhat7).T; pred7=np.mat(pred7).T



		stacking_train=np.hstack((yhat1,yhat2,yhat3,yhat4,yhat6,yhat7))  ###  data
		stacking_label=train_y
		stacking_test=np.hstack((pred1,pred2,pred3,pred4,pred6,pred7))
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
	resultDF.to_csv(fileName+'_AM_predict_stacking_07model.csv')
	return 0#xgbMape#,sumScore


















