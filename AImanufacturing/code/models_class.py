#############################################
###
###  我要写一个所有model的训练类
##############################################
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.svm import SVR 
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
import numpy as np
#from sklearn.ensemble import GradientBoostingRegressor
from pandas import Series,DataFrame
#from sklearn.model_selection import train_test_split
#from sklearn import datasets
from sklearn import cross_validation
#import tensorflow as tf 
#import skflow
import xgboost as xgb
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier


class FuckingTrainingClass:
	def __init__(self,fileName):
		self.param={
		'booster':'gbtree',
		'objective': 'reg:linear', 
		#'n_estimators':100,
		#'gamma':0.1, 
		'max_depth':6, 
		'lambda':0.1, 
		#'alpha':10,
		'subsample':0.8, # 采样训练样本
		'colsample_bytree':0.9, # 列采样
		'min_child_weight':1, 
		'silent':1 ,
		'eta': 0.02, # 学习率
		#'seed':100,
		#'nthread':7,
		#'eval_metric': 'auc'
		}



	def load_data(filename):
		data=pd.read_csv(fileName)
		return data

	def data_pre():
		pass

	def cross_validation():
		pass

	def data_to_csv():
		pass

	def mse(yact,yhat):
		return ((yhat-yact)**2).mean()

	def XGB():
		pass

	def GBDT():
		pass

	def logistc():
		pass

	def SVM():
		pass












