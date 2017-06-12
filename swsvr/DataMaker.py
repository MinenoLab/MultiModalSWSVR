import sys
import time
import datetime
import numpy as np
from DataStructure import SdcData

def convert_unixtime(str):
    return int(time.mktime(str.timetuple()))

def convert_datetime(unixtime):
    return datetime.datetime.fromtimestamp(unixtime)

def convertWithFormat(str):
    newstr = str[0:4]+"-"+str[4:6]+"-"+str[6:8]+" "+str[8:10]+":"+str[10:12]
    return datetime.datetime.strptime(newstr, '%Y-%m-%d %H:%M')

def advance(str, min=0):
    newstr = convertWithFormat(str)
    unixtime = convert_unixtime(newstr)
    unixtime += min * 60
    return convert_datetime(unixtime)

def reverse_advance(str, min=0):
    newstr = convertWithFormat(str)
    unixtime = convert_unixtime(newstr)
    unixtime -= min * 60
    return convert_datetime(unixtime)

def datetime2org(dt):
    def one2two(num):
        if len(str(num)) == 1:
            return "0"+str(num)
        else:
            return str(num)
    org_dt = one2two(dt.year) + one2two(dt.month) + one2two(dt.day) + one2two(dt.hour) + one2two(dt.minute)
    return org_dt

def convert2SDC(X, y, sensors, state, area, dateAndTime, ph):
    pre_X=[]
    pre_y=[]
    fol_X=[]
    pre_XS=[]
    fol_XS=[]
    area_sdc = []
    dateAndTime_sdc = []
    dateAndTimeCon=np.array([convertWithFormat(tmp_dt) for tmp_dt in dateAndTime])
    state=np.array(state)
    area=np.array(area)
    for var1, var2, sd, st, ar, dt in zip(X, y, sensors, state, area, dateAndTime):
        elapsed_time = advance(dt, ph)
        state_w = np.where(state==st)
        area_w = np.where(area==ar)
        dt_w = np.where(dateAndTimeCon==elapsed_time)
        
        state_set=set(list(state_w[0]))
        area_set=set(list(area_w[0]))
        dt_set=set(list(dt_w[0]))
        matched_set=list(state_set & area_set & dt_set)
        if len(matched_set)==1: 
                dateAndTime_sdc.append(dt)
                area_sdc.append(ar)
                pre_X.append(var1)
                pre_y.append(y[matched_set[0]])
                fol_X.append(X[matched_set[0]])
                pre_XS.append(sd)
                fol_XS.append(sensors[matched_set[0]])
                print str(matched_set[0])+":"+str(dt)+"->"+str(elapsed_time)
    pre_X = np.array(pre_X)
    pre_y = np.array(pre_y)
    fol_X = np.array(fol_X)
    pre_XS = np.array(pre_XS)
    fol_XS = np.array(fol_XS)
    
    return SdcData(pre_X, pre_y, fol_X, pre_XS, fol_XS), area_sdc, dateAndTime_sdc