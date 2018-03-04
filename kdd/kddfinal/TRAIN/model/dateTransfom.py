###########################################################
#   不搞了！
#
###########################################################
import numpy as np
import pandas as pd 
from datetime import datetime,timedelta

fileName='submission_phase2_travelTime.csv'

def timeTrans(fileName):

	fr=open(fileName,'r')
	fr.readline()
	time_data=fr.readlines()
	fr.close()
	print(time_data[0])

	for i in range(len(time_data)):
		each_window=time_data[i].replace('"', '').split(',') #l两个值，str型


		timestr1=each_window[2].split('[')[1]
		timestr2=each_window[3].split(')')[0]

		t1=datetime.strptime(timestr1,'%Y-%m-%d %H:%M:%S')
		t2=datetime.strptime(timestr2,'%Y-%m-%d %H:%M:%S')

		t01=t1+timedelta(days=7)
		t02=t2+timedelta(days=7)

		str1=str(t01); str2=str(t02)




	

    fw = open(fileName, 'w')
    fw.writelines('"time_window"')


    for route in travel_times.keys():
        route_time_windows = list(travel_times[route].keys())
        route_time_windows.sort()
        for time_window_start in route_time_windows:
            time_window_end = time_window_start + timedelta(minutes=20)
            tt_set = travel_times[route][time_window_start]
            avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)

            #### 修改了输出格式   ########################
            out_line = ','.join(['"' + route + '"',
                                 '"' + str(time_window_start) + '"',
                                 '"' + str(avg_tt) + '"']) + '\n'
            fw.writelines(out_line)
    fw.close()
