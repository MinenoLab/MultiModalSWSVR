import os
from chainer import cuda
from plants3d import PLANTS
from swsvr.DataMaker import convert2SDC
try: import cPickle as pickle
except: import pickle
import numpy as np
import joblib
from sklearn.preprocessing  import StandardScaler

try:
    cuda.check_cuda_available()
    xp = cuda.cupy
except:
    xp = np

def format_for_chainer(X, resized_tuple, is_color, target):
        if is_color: X = np.reshape(X, (X.shape[0], 3*len(target), resized_tuple[0], resized_tuple[1]))
        else: X = np.reshape(X, (X.shape[0], 1*len(target), resized_tuple[0], resized_tuple[1]))
        X = X.astype(xp.float32)
        return X
    
def load_data(dp, IS_SWSVR=False):
    path_input_dir = dp.PATH_INPUT_DIR
    resized_tuple = dp.RESIZED_TUPLE
    is_color = dp.IS_COLOR
    is_gcn = dp.IS_GCN
    is_data_arg = dp.IS_DATA_ARG
    ph = dp.ph
    is_zca = dp.IS_ZCA
    is_rsd=dp.IS_RSD
    start=dp.start
    end=dp.end
    area=dp.area
    target = dp.target
    method = dp.method
    obj_area = dp.obj_area
    thinNum = dp.thinNum
    tddataThin = dp.tddataThin
    
    # init #
    pname = "Tuple"+str(locals()['resized_tuple'])+"_Color"+str(locals()['is_color'])[0] \
                   + "_Gcn"+str(locals()['is_gcn'])[0]+"_Arg"+str(locals()['is_data_arg'])[0] \
                   +"_Ph"+str(locals()['ph'])[0] + "_Zca"+str(locals()['is_zca'])[0] \
                   +"_Rsd"+str(locals()['is_rsd'])+"_Start"+str(locals()['start']) \
                   +"_End"+str(locals()['end'])+"_Area"+str(locals()['area']) \
                   +"_"+str(locals()['target']) +"_M"+str(locals()['method']) +"_OA"+str(locals()['obj_area']) \
                   + "_Thin"+str(locals()['thinNum'])+"_tddataThin"+str(locals()['tddataThin'])
    print " + PROJECT_NAME:"+pname
    PATH_TO_DATA_DIR=path_input_dir+"/data"
    if os.path.exists(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname)))==False:
        os.mkdir(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname)))
    if os.path.exists(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, "tddata.pkl")))==False or os.path.exists(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, "testdata.pkl")))==False:    
        IS_PRE = True
    else:
        IS_PRE = False

    if IS_PRE==True:# make new datasets #
        print " + Loading training data"
        tddata = PLANTS(which_set= "train", path=path_input_dir, pname=pname, PATH_TO_DATA_DIR=PATH_TO_DATA_DIR,
                       resized_tuple=resized_tuple, 
                       is_color = is_color,
                       is_gcn=is_gcn,
                       is_data_arg=is_data_arg,
                       is_rsd=is_rsd,
                       start=start,
                       end=end,
                       inArea=area,
                       target = target,
                       method = method,
                       obj_area=obj_area,
                       thinNum = thinNum
                       )
        if is_zca:
            preprocessor = preprocessing.ZCA()
            tddata.apply_preprocessor(preprocessor=preprocessor, can_fit=True)
        joblib.dump(tddata, os.path.join(PATH_TO_DATA_DIR, pname, 'tddata.pkl'), compress=3)
        if ph!=None:
            tddata_sdc, train_ar, train_dt = convert2SDC(tddata.X, tddata.y, tddata.sensors_data, tddata.state, tddata.area, tddata.dateAndTime, ph)
            joblib.dump(tddata_sdc, os.path.join(PATH_TO_DATA_DIR, pname, 'tddata_sdc_%d.pkl'%ph), compress=3)   
            joblib.dump(train_ar, os.path.join(PATH_TO_DATA_DIR, pname, 'tddata_sdc_ar_%d.pkl'%ph), compress=3)   
            joblib.dump(train_dt, os.path.join(PATH_TO_DATA_DIR, pname, 'tddata_sdc_dt_%d.pkl'%ph), compress=3)   
                  
        print " + Loading testing data"
        testdata = PLANTS(which_set= "test", path=path_input_dir, pname=pname, PATH_TO_DATA_DIR=PATH_TO_DATA_DIR,
                          resized_tuple=resized_tuple,
                          is_color = is_color,
                          is_gcn=is_gcn,
                          is_data_arg = is_data_arg,
                          is_rsd=is_rsd,
                          start=start,
                          end=end,
                          inArea=area,
                          target = target,
                          method = method,
                          obj_area=obj_area,
                          thinNum = thinNum
                          )
        if is_zca:
            testdata.apply_preprocessor(preprocessor=preprocessor, can_fit=False)
        joblib.dump(testdata, os.path.join(PATH_TO_DATA_DIR, pname, 'testdata.pkl'), compress=3)
        if ph!=None:
            testdata_sdc, test_ar, test_dt = convert2SDC(testdata.X, testdata.y, testdata.sensors_data, testdata.state, testdata.area, testdata.dateAndTime, ph)
            joblib.dump(testdata_sdc, os.path.join(PATH_TO_DATA_DIR, pname, 'testdata_sdc_%d.pkl'%ph), compress=3)
            joblib.dump(test_ar, os.path.join(PATH_TO_DATA_DIR, pname, 'testdata_sdc_ar_%d.pkl'%ph), compress=3)
            joblib.dump(test_dt, os.path.join(PATH_TO_DATA_DIR, pname, 'testdata_sdc_dt_%d.pkl'%ph), compress=3)
            
    else:
        print " + Loading saved data"
        tddata = joblib.load(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, 'tddata.pkl')))
        testdata = joblib.load(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, 'testdata.pkl')))
        if ph!=None:
            print " + Loading saved data for SDC"
            tddata_sdc = joblib.load(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, 'tddata_sdc_%d.pkl'%ph)))
            train_ar = joblib.load(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, 'tddata_sdc_ar_%d.pkl'%ph)))
            train_dt = joblib.load(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, 'tddata_sdc_dt_%d.pkl'%ph)))
            testdata_sdc = joblib.load(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, 'testdata_sdc_%d.pkl'%ph)))
            test_ar = joblib.load(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, 'testdata_sdc_ar_%d.pkl'%ph)))
            test_dt = joblib.load(os.path.abspath(os.path.join(PATH_TO_DATA_DIR, pname, 'testdata_sdc_dt_%d.pkl'%ph)))
    
    if ph==None:
        X_td=tddata.X
        y_td=tddata.y
        X_test=testdata.X
        y_test=testdata.y
        Xs_td=tddata.sensors_data
        Xs_test=testdata.sensors_data
        X_td_fol = None
        X_test_fol = None
        Xs_td_fol = None
        Xs_test_fol = None
        train_ar=tddata.area
        train_dt=tddata.dateAndTime
        test_ar=testdata.area
        test_dt=testdata.dateAndTime
    else:
        X_td=tddata_sdc.pre_X
        y_td=tddata_sdc.pre_y
        X_test=testdata_sdc.pre_X
        y_test=testdata_sdc.pre_y
        Xs_td=tddata_sdc.pre_XS
        Xs_test=testdata_sdc.pre_XS
        X_td_fol = tddata_sdc.fol_X
        X_test_fol = testdata_sdc.fol_X
        Xs_td_fol = tddata_sdc.fol_XS
        Xs_test_fol = testdata_sdc.fol_XS
    
    y_td = y_td.astype(xp.float32)
    y_test = y_test.astype(xp.float32)

    X_td = format_for_chainer(X_td, resized_tuple, is_color, target)
    X_test = format_for_chainer(X_test, resized_tuple, is_color, target)
    if X_test_fol is not None: X_td_fol = format_for_chainer(X_td_fol, resized_tuple, is_color, target)
    if X_test_fol is not None: X_test_fol = format_for_chainer(X_test_fol, resized_tuple, is_color, target)
    
    scaler = StandardScaler()
    scaler.fit(Xs_td)
    Xs_td = scaler.transform(Xs_td).astype(xp.float32)
    Xs_test = scaler.transform(Xs_test).astype(xp.float32)
    if Xs_td_fol is not None: Xs_td_fol = scaler.transform(Xs_td_fol).astype(xp.float32)
    if Xs_test_fol is not None: Xs_test_fol = scaler.transform(Xs_test_fol).astype(xp.float32)
    
    if not is_gcn:
        print "<!> new norm!!"
        mean_td = Xs_td.mean()
        var_td = Xs_td.var(ddof=1)
        Xs_td = (Xs_td -mean_td)/var_td
        Xs_test = (Xs_test -mean_td)/var_td
        if Xs_td_fol is not None: Xs_td_fol = (Xs_td_fol -mean_td)/var_td
        if Xs_test_fol is not None: Xs_test_fol = (Xs_test_fol -mean_td)/var_td
    
    print " + The training data (size, features): "+str(X_td.shape)
    print " + The testing data  (size, features): "+str(X_test.shape)
    
    if IS_SWSVR:
        return X_td, X_td_fol, Xs_td, Xs_td_fol, y_td, \
               X_test, X_test_fol, Xs_test, Xs_test_fol, y_test, \
               train_ar, train_dt, test_ar, test_dt
    else:
        return pname, X_td, y_td, X_test, y_test, Xs_td, Xs_test, train_ar, train_dt, test_ar, test_dt