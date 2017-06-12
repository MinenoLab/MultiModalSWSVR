import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def get_mae(y_true, y_pred):
    if len(y_true.shape) == 2:
        y_true = y_true.reshape(np.max((y_true.shape[0],y_true.shape[1])))
    if len(y_pred.shape) == 2:
        y_pred = y_pred.reshape(np.max((y_pred.shape[0],y_pred.shape[1])))
        
    return mean_absolute_error(y_true, y_pred)

def get_mape(y_true, y_pred):
    if len(y_true.shape) == 2:
        y_true = y_true.reshape(np.max((y_true.shape[0],y_true.shape[1])))
    if len(y_pred.shape) == 2:
        y_pred = y_pred.reshape(np.max((y_pred.shape[0],y_pred.shape[1])))
    
    y_pred = np.delete(y_pred, np.where(y_true==0), 0)
    y_true = np.delete(y_true, np.where(y_true==0), 0)
    tmp_mape = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(tmp_mape) * 100
    return mape

def get_rmse(y_true, y_pred):
    if len(y_true.shape) == 2:
        y_true = y_true.reshape(np.max((y_true.shape[0],y_true.shape[1])))
    if len(y_pred.shape) == 2:        
        y_pred = y_pred.reshape(np.max((y_pred.shape[0],y_pred.shape[1])))
        
    return math.sqrt(mean_squared_error(y_true, y_pred))

def get_mse(y_true, y_pred):
    if len(y_true.shape) == 2:
        y_true = y_true.reshape(np.max((y_true.shape[0],y_true.shape[1])))
    if len(y_pred.shape) == 2:        
        y_pred = y_pred.reshape(np.max((y_pred.shape[0],y_pred.shape[1])))
        
    return mean_squared_error(y_true, y_pred)

def get_rse(y_true, y_pred):
    if len(y_true.shape) == 2:
        y_true = y_true.reshape(np.max((y_true.shape[0],y_true.shape[1])))
    if len(y_pred.shape) == 2:        
        y_pred = y_pred.reshape(np.max((y_pred.shape[0],y_pred.shape[1])))
    
    rmse2 = (get_rmse(y_true, y_pred) ** 2) * len(y_true)
    rmse2_ave = (get_rmse(y_true, np.full(len(y_true),np.average(y_true))) ** 2) * len(y_true)
    
    return rmse2/rmse2_ave

def get_rae(y_true, y_pred):
    if len(y_true.shape) == 2:
        y_true = y_true.reshape(np.max((y_true.shape[0],y_true.shape[1])))
    if len(y_pred.shape) == 2:        
        y_pred = y_pred.reshape(np.max((y_pred.shape[0],y_pred.shape[1])))
    
    mae2 = (get_mae(y_true, y_pred)) * len(y_true)
    mae2_ave = (get_mae(y_true, np.full(len(y_true),np.average(y_true)))) * len(y_true)
    
    return mae2/mae2_ave

def metrics(y_true, y_pred):
    return get_mae(y_true, y_pred), get_rmse(y_true, y_pred), get_mape(y_true, y_pred)
