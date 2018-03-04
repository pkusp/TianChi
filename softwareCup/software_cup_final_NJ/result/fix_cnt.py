import numpy as np
import pandas as pd
from pandas import Series,DataFrame


dd=pd.read_csv('data_submit_with_dayid_0904.csv')


dd=dd[dd['round']>300]

dd['cnt']=dd['round']/800

dd['cnt']=dd['cnt']//1

xx=dd['cnt'].replace(0,1)

dd['cnt']=xx

dd=dd.drop('Unnamed: 0',axis=1)
dd.to_csv('final_data_submit_with_dayid_0904.csv')

