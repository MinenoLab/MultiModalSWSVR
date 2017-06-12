'''
Created on Nov 28, 2016

@author: minelab
'''
from chainer import cuda

from load import load_data_old

try: import cPickle as pickle
except: import pickle
import numpy as np
from utility.metrics import get_mae
from swsvr.DataStructure import SdcData
from train import init_data_params, evaluate, sampling_hyperparameters, init_args
from swsvr.Tuner import grid_tuning
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import joblib
import os
from swsvr.Tuner import Errors
from plants3d import sep_kaika


# sample parameters to tune

swsvr_params = {'svr_cost': [0.001, 0.01, 0.1, 1, 8, 32, 128],
                    'svr_epsilon': [0.00001],
                    'svr_intercept': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                    'svr_itr':[100000],
                    'kapp_gamma': [0.00001], #unused
                    'kapp_num': [100], #unused
                    'pls_compnum': [50], #unused
                    'sdc_weight': [0.5, 1, 3],
                    'predict_weight': [0.5, 1, 3],
                    'lower_limit': [10],
                    'n_estimators':[10, 50, 100]}
"""
swsvr_params = {'svr_cost': [2, 3],
                    'svr_epsilon': [0.00011],
                    'svr_intercept': [1],
                    'svr_itr':[100000],
                    'kapp_gamma': [0.00001],
                    'kapp_num': [100],
                    'pls_compnum': [50],
                    'sdc_weight': [0.5, 1, 3],
                    'predict_weight': [0.5],
                    'lower_limit': [10],
                    'n_estimators':[5]}
"""
def visualization(true_y, swsvr_result, other, dnn, savepath):
    name=[]
    mape=[]
    er = Errors()
    print "+ Each prediction error for MAPE"
    true_y = true_y.reshape(np.max((true_y.shape[0],true_y.shape[1])))
    
    print "SWSVR: ",
    name.append("swsvr")
    mape.append(get_mae(true_y, swsvr_result))
    er.set_errors(true_y, swsvr_result, None)
    er.print_now()

    print "DNN: ",
    name.append("dnn")
    mape.append(get_mae(true_y, dnn))
    er.set_errors(true_y, dnn, None)
    er.print_now()

    for l in simple_learners:
        name.append(l.name)
        print "%s: " % l.name,
        tmp_mape = get_mae(true_y, l.result)
        mape.append(tmp_mape)
        er.set_errors(true_y, l.result, None)
        er.print_now()  
    
    #save 
    df = pd.DataFrame({'true':true_y, 'swsvr':swsvr_result})
    df = pd.concat([df,pd.DataFrame({'dnn':dnn.reshape(dnn.shape[0])})],axis=1)
    for l in other:
        if len(l.result.shape) == 2:
            l.result = l.result.reshape(l.result.shape[0])
        tmp = pd.DataFrame({l.name:l.result})
        df = pd.concat([df,tmp],axis=1)
    df.to_csv(savepath+"detail.csv")

    plt.figure(figsize=(12,8))
    plt.subplot2grid((2, 1), (0, 0))
    sea.barplot(x=name,y=mape)
    plt.axhline(y=mape[0], color='r', ls=':')
    plt.title("MAPE")
    plt.subplot2grid((2, 1), (1, 0))
    plt.xlim(0, len(true_y))
    for l in other:
        plt.plot(l.result,lw=1,alpha=0.3,label=l.name)
    plt.plot(dnn, "green", lw=3, label='dnn')
    plt.plot(swsvr_result,"red",lw=3,label='swsvr')
    plt.plot(true_y,"blue",lw=4,label='true')
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='center', borderaxespad=0, ncol=10)
    plt.title("Regression curve")
    plt.show()

def mapping(hps, rnn, x, sensors, args):
    all_weights = np.empty((0, 64))
    rnn = rnn.copy()
    rnn.reset_state()
    for i in range(0, x.shape[0], hps.batch_size):
        # make batch
        x_batch = cuda.to_gpu(np.array(x[i:i + hps.batch_size]))
        if args.is_sensor:
            sensors_batch = cuda.to_gpu(np.array(sensors[i:i + hps.batch_size]))
            weights = rnn.get_weights(hps, x_batch, sensors_batch)
        else:
            weights = rnn.get_weights(hps, x_batch)
        all_weights = np.vstack((all_weights, weights))
    
    return all_weights

if __name__ == '__main__':
    # init params
    IS_BUILD = False
    # tdthin1 #TODO TRAIN KAERO!
    org_model_path = "/home/minelab/git/timeseries_wilt_prediction/result/Tuple(144, 144)_ColorT_GcnT_ArgF_Ph6_ZcaF_RsdFalse_Start0_End0_Area[1, 2, 3, 4]_['rm_cleaned_norm_masked_img']_Marea_OA[2]_Rate0.3_Thin1_tddataThin1/lr9.71278442547e-05_bs99_ln1_ru64_sl1_do0.436352322041_RMSpropGraves_e300/models/"
    # tdthin2 #TODO TRAIN KAERO!
    #org_model_path = "/home/minelab/git/timeseries_wilt_prediction/result/Tuple(144, 144)_ColorT_GcnT_ArgF_Ph6_ZcaF_RsdFalse_Start0_End0_Area[1, 2, 3, 4]_['rm_cleaned_norm_masked_img']_Marea_OA[2]_Rate0.3_Thin1_tddataThin2/lr1.18355957657e-06_bs49_ln1_ru64_sl1_do0.59248570844_RMSpropGraves_e300/models/"
    # tdthin3 #TODO TRAIN KAERO!
    #org_model_path = "/home/minelab/git/timeseries_wilt_prediction/result/Tuple(144, 144)_ColorT_GcnT_ArgF_Ph6_ZcaF_RsdFalse_Start0_End0_Area[1, 2, 3, 4]_['rm_cleaned_norm_masked_img']_Marea_OA[2]_Rate0.3_Thin1_tddataThin3/lr3.8456489571e-06_bs88_ln1_ru64_sl1_do0.552105200435_RMSpropGraves_e300/models/"
    # 4
    #org_model_path = "/home/minelab/git/timeseries_wilt_prediction/result/Tuple(144, 144)_ColorT_GcnT_ArgF_Ph6_ZcaF_RsdFalse_Start0_End0_Area[1, 2, 3, 4]_['rm_cleaned_norm_masked_img']_Marea_OA[2]_Rate0.3_Thin1_tddataThin4/lr1.29080652856e-06_bs55_ln1_ru64_sl1_do0.668782470674_RMSpropGraves_e300/models/"
    # 10
    #org_model_path = "/home/minelab/git/timeseries_wilt_prediction/result/Tuple(144, 144)_ColorT_GcnT_ArgF_Ph6_ZcaF_RsdFalse_Start0_End0_Area[1, 2, 3, 4]_['rm_cleaned_norm_masked_img']_Marea_OA[2]_Rate0.3_Thin1_tddataThin10/lr1.30860380257e-06_bs14_ln1_ru64_sl1_do0.45122378369_RMSpropGraves_e300/models/"
    # 20
    #org_model_path = "/home/minelab/git/timeseries_wilt_prediction/result/Tuple(144, 144)_ColorT_GcnT_ArgF_Ph6_ZcaF_RsdFalse_Start0_End0_Area[1, 2, 3, 4]_['rm_cleaned_norm_masked_img']_Marea_OA[2]_Rate0.3_Thin1_tddataThin20/lr7.87202348231e-06_bs40_ln1_ru64_sl1_do0.591289430548_RMSpropGraves_e300/models/"
    
    # load data
    # initialize params    
    args = init_args()
    dp = init_data_params()
    X_td, X_td_fol, Xs_td, Xs_td_fol, y_td, X_test, X_test_fol, Xs_test, Xs_test_fol, y_test, train_ar, train_dt, test_ar, test_dt = load_data_old(dp, IS_SWSVR=True)
    
    # thin tddata
    tmp_x_train = []
    tmp_x_train_f = []
    tmp_y_train = []
    tmp_train_sensors = []
    tmp_train_sensors_f = []
    for i in range(len(X_td)):
        if i % dp.tddataThin == 0:
            tmp_x_train.append(X_td[i])
            tmp_x_train_f.append(X_td_fol[i])
            tmp_y_train.append(y_td[i])
            tmp_train_sensors.append(Xs_td[i])
            tmp_train_sensors_f.append(Xs_td_fol[i])
    X_td = np.array(tmp_x_train)
    X_td_fol = np.array(tmp_x_train_f)
    y_td = np.array(tmp_y_train)
    Xs_td = np.array(tmp_train_sensors)
    Xs_td_fol = np.array(tmp_train_sensors_f)
    
    x_valid1, valid_sensors1, y_valid1, x_test1, test_sensors1, y_test1 = sep_kaika(dp, test_ar, test_dt, X_test, Xs_test, y_test, target_num = 0)
    
    # debug
    """
    X_td = X_td[0:(len(X_td) / 40) * 40]
    Xs_td = Xs_td[0:(len(Xs_td) / 40) * 40]
    y_td = y_td[0:(len(y_td) / 40) * 40]
    """
    
    # load trained dnn model    
    model_path = org_model_path + "latest.chainermodel"
    rnn = pickle.load(open(model_path))
    
    # predict train data by using trained dnn only for debug
    train_errors = Errors()
    hps = sampling_hyperparameters(True)
    py = evaluate(hps, rnn, X_td, Xs_td, y_td, args.is_sensor)
    train_errors.set_errors(y_td, py, None)
    train_errors.print_now()

    # convert sdc data    
    tddata_sdc_rnn = SdcData(mapping(hps, rnn, X_td, Xs_td, args),
                             y_td,
                             mapping(hps, rnn, X_td_fol, Xs_td_fol, args),)
    testdata_sdc_rnn = SdcData(mapping(hps, rnn, X_test, Xs_test, args),
                             y_test,
                             mapping(hps, rnn, X_test_fol, Xs_test_fol, args),)
    validdata_sdc_rnn, testdata_sdc_rnn = testdata_sdc_rnn.sep_kaika_sdc(dp, test_ar, test_dt, target_num= 0)
    print "train data:%s valid data:%s test data%s"%(str(tddata_sdc_rnn.pre_X.shape), str(validdata_sdc_rnn.pre_X.shape), str(testdata_sdc_rnn.pre_X.shape))

    if IS_BUILD:
        # buid model
        swsvr=grid_tuning(swsvr_params, tddata_sdc_rnn, validdata_sdc_rnn, testdata_sdc_rnn)
        #simple_learners = other_learners_tuning(tddata_sdc_rnn, validdata_sdc_rnn, testdata_sdc_rnn)
        # model save
        joblib.dump(swsvr, os.path.join(org_model_path, 'swsvr.pkl'), compress=3)
        #joblib.dump(simple_learners, os.path.join(org_model_path, 'others.pkl'), compress=3)
    else:
        # load trained model
        swsvr = joblib.load(os.path.join(org_model_path, 'swsvr.pkl'))
        simple_learners = joblib.load(os.path.join(org_model_path, 'others.pkl'))
    
    # predict test data
    sr, true_y = swsvr.predict(testdata_sdc_rnn.pre_X, varbose=True), testdata_sdc_rnn.pre_y
    swsvr_result=sr[0]
    weight_vector=sr[1]
    each_result=sr[2]
    #for i in range(len(each_result)):
    for i in range(10):
        #if abs(np.corrcoef(true_y.reshape(4607), weight_vector[i])[0,1])<0.4: continue
        #if np.corrcoef(true_y.reshape(4607), weight_vector[i])[0,1]>-0.2 or np.corrcoef(true_y.reshape(4607), weight_vector[i])[0,1]<-0.4: continue
        fig,ax1=plt.subplots(); ax2=ax1.twinx(); plt.xlim([0,4603]);ax1.plot(true_y,"k",lw=3);ax2.set_ylim(0.3,0.8)
        ax1.plot(each_result[i],"red",lw=1, alpha=0.7)
        ax2.plot(weight_vector[i], "c", lw=1, alpha=0.7)
        ppt = "%s,%s,%s,%s"%(str(i),str(np.corrcoef(true_y.reshape(4607), weight_vector[i])[0,1]), str(swsvr.weak_learners[i].train_y.shape[0]), np.average(swsvr.weak_learners[i].train_y))
        plt.title(ppt)
        #np.savetxt("restxt.csv", each_result[79], fmt="%0.5f", delimiter=",")
    for l in simple_learners: l.predict(testdata_sdc_rnn.pre_X)
    
    # visualize for result
    visualization(true_y, swsvr_result, simple_learners, evaluate(hps, rnn, x_test1, test_sensors1, y_test1, args.is_sensor), org_model_path)
