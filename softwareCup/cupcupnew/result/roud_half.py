import numpy as np
import pandas as pd
from pandas import Series,DataFrame


dd=pd.read_csv('final_data_submit_with_dayid_0904.csv')

dd['round']=dd['round']//150*100

dd=dd.drop('Unnamed: 0',axis=1)
dd.to_csv('half_data_submit_with_dayid_0904.csv')
print(dd.head())