import numpy as np
from numba.decorators import jit

@jit('f8(f8[:],f8[:])')
def cal_weighted_average(vector, weight_vector):
    return np.average(vector,weights=weight_vector)

@jit('f8(f8[:],f8[:])')
def calc_euclid(A, B):
    '''
    Args:
        features: numpy
    '''
    return np.linalg.norm(A - B)

@jit('f8[:](f8[:],f8)')
def calc_reciprocal(double, weight_param):
    '''
    This method calc weight from a double value.
    '''
    denominator = np.power(double, weight_param)
    index = np.where(denominator == 0)[0]
    for i in index:# double(arg1) has some zero
        denominator[i] = np.min(denominator[np.where(denominator>0)])
    denominator = np.reciprocal(denominator)

    return denominator

@jit('f8[:](f8[:,:],f8[:,:])')
def calc_euclid_matrix(from_data, to_data):
    '''
    This method calculates euclid distances for matrix.
    Parameters must be FeaturesData class.
    '''
    return np.linalg.norm((from_data - to_data), axis=1)