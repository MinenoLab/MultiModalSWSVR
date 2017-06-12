'''
Created on 2015/10/05

@author: Kaneda
'''

import numpy as np
import swsvr.Calculator as ut
import swsvr.DataStructure as ds
from numba.decorators import jit

def __init__(self):
    '''
    Constructor
    '''

@jit('f8[:](f8[:],f8)')
def calc_weight(double, weight_param):
    '''
    This method calc weight from a double value.
    '''
    return ut.calc_reciprocal(double, weight_param)

def calc_radius(sdc_vector, weight_vector):
    '''
    This method calculates radius for SDC.
    '''
    return ut.cal_weighted_average(sdc_vector,weight_vector)

def extract(distance_G_vector, sdc_vector, sdc_data, param_info):
    '''
    This method extracts training data for SDC.
    '''
    weight_vector = calc_weight(distance_G_vector, param_info.sdc_weight)
    r = calc_radius(sdc_vector, weight_vector)
    condition = np.where(distance_G_vector < r)
    if condition[0].size < param_info.lower_limit:#SSDC
        '''
        r = np.std(distance_G_vector)
        condition = np.where(distance_G_vector < r)
        print "SSDC size :",condition[0].size
        '''
        condition = np.argsort(distance_G_vector)[1:1001]
        #print "SSDC size :",condition.size
    else:#DSDC
        #print "DSDC size :",condition[0].size
        pass
    return ds.LabeledData(sdc_data.pre_X[condition], sdc_data.pre_y[condition])
