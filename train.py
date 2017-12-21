# coding: utf-8
import csv
import os

import numpy as np
import six

try: import cPickle as pickle
except: import pickle
import random
import argparse
import matplotlib.pyplot as plt
from collections import namedtuple
from load import load_data
from plants3d import sep_kaika
from visualize.myplot import init_plot
from swsvr.Tuner import Errors
from chainer import cuda, optimizers

from model.CNN_4f2 import CNN_4f2
from model.CNN_42 import CNN_42

def init_args():
    # arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_sensor', type=str, default=True)
    parser.add_argument('--is_init_model', type=str, default=True)
    parser.add_argument('--tuned_model', type=str, default='./tuned_model/model.pkl')
    parser.add_argument('--out_path', type=str, default='./result')

    # setting parameters
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--learning_rate_decay', type=float, default=0.97)
    parser.add_argument('--learning_rate_decay_after', type=int, default=10)
    parser.add_argument('--decay_rate', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=int, default=5)
    
    return parser.parse_args()

def sampling_hyperparameters():
    HyperParameter = namedtuple('HyperParameter', 'learning_rate batch_size layer_num rnn_units seq_length dropout optimizer')
    
    all_param = {
        "learning_rate" : [random.uniform(-6, -4)],
        #"batch_size" : [random.randint(10,100)],
        "batch_size": [5],
        "layer_num" : [1], 
        "rnn_units" : [64],
        "seq_length" : [1],
        "dropout" : [random.uniform(0.3, 0.7)],
        "optimizer" : ['RMSpropGraves']
    }

    learning_rate = 10 ** random.choice(all_param["learning_rate"])
    batch_size = random.choice(all_param["batch_size"])
    layer_num = 1
    rnn_units = random.choice(all_param["rnn_units"])
    seq_length = random.choice(all_param["seq_length"])
    dropout = random.choice(all_param["dropout"])
    optimizer = random.choice(all_param["optimizer"])

    return HyperParameter(learning_rate, batch_size, layer_num, rnn_units, seq_length, dropout, optimizer)

def init_data_params():
    data_params = namedtuple('data_params', 'PATH_INPUT_DIR RESIZED_TUPLE IS_COLOR IS_GCN IS_DATA_ARG ph IS_ZCA IS_RSD start end area target method obj_area thinNum tddataThin')
    PATH_INPUT_DIR = "dataset"
    RESIZED_TUPLE = (144, 144)
    IS_COLOR=True
    IS_GCN=True
    IS_DATA_ARG=False
    IS_ZCA=False
    IS_RSD=False
    ph=60
    start=0 #yyyymmddhhss
    end=0 #yyyymmddhhss
    area=[1, 2] # number or "all"
    target = ["ROAF"] # pic or ROAF or POF
    method = "area" #area or th or kaika
    obj_area= [2] # validation and test area
    thinNum = 1 # thin for train data
    tddataThin = 1 # thin for validation and test data
    
    dp = data_params(PATH_INPUT_DIR, RESIZED_TUPLE, IS_COLOR, IS_GCN, IS_DATA_ARG, ph, IS_ZCA, IS_RSD, start, end, area, target, method, obj_area, thinNum, tddataThin)
    return dp

def evaluate(hps, model, x, sensors, y, is_sensor):
    pys = np.empty((0, 1))
    model = model.copy()
    model.reset_state()
    for i in range(0, x.shape[0], hps.batch_size):
        # make batch
        x_batch = cuda.to_gpu(np.array(x[i:i + hps.batch_size]))
        sensors_batch = cuda.to_gpu(np.array(sensors[i:i + hps.batch_size]))
        y_batch = cuda.to_gpu(np.array(y[i:i + hps.batch_size]))

        # forward and calculate loss value
        if is_sensor:
            py, loss_i = model.forward_one_step(hps, x_batch, sensors_batch, y_batch, train=False)
        else:
            py, loss_i = model.forward_one_step(hps, x_batch, y_batch, train=False)
        pys = np.vstack((pys, py.data.get()))
    
    return pys

def select_optimizer(opt_name, learning_rate):
    if opt_name == "Adam":
        return optimizers.Adam(alpha=learning_rate)
    elif opt_name == "SGD":
        return optimizers.SGD(lr=learning_rate)
    elif opt_name == "RMSpropGraves":
        return optimizers.RMSpropGraves(lr=learning_rate)
    elif opt_name == "RMSprop":
        return optimizers.RMSprop(lr=learning_rate)
    elif opt_name == "AdaDelta":
        return optimizers.AdaDelta()
    elif opt_name == "AdaGrad":
        return optimizers.AdaGrad(lr=learning_rate)
    elif opt_name == "MomentumSGD":
        return optimizers.MomentumSGD(lr=learning_rate)
    elif opt_name == "NesterovAG":
        return optimizers.NesterovAG(lr=learning_rate)
    else:
        print('please select correct optimizer')
        exit()
     
def main(dp, hps, args, root_dir, x_train, y_train, x_valid, y_valid, train_sensors, valid_sensors, valid_ar, valid_dt):
    
    # if hyperparams.csv does not have header, write header based on keys of hps
    errors_csv = os.path.join(root_dir, 'errors.csv')
    if not os.path.exists(errors_csv):
        with open(errors_csv, 'a+') as f:
            writer = csv.writer(f)
            header = vars(hps).keys() + ["mae_td","rmse_td","mape_td","mse_td","rse_td","rae_td",
                                         "mae_valid","rmse_valid","mape_valid","mse_valid","rse_valid","rae_valid",
                                         "mae_test","rmse_test","mape_test","mse_test","rse_test","rae_test"
                                         ]
            writer.writerow(header)
    
    # make directory for   save some result at each parameter
    project = "lr" + str(hps.learning_rate) + "_bs" + str(hps.batch_size) + \
              "_ln" + str(hps.layer_num) + "_ru" + str(hps.rnn_units) + "_sl" + str(hps.seq_length) + "_do" + str(hps.dropout) + "_" + str(hps.optimizer) + "_e" + str(args.n_epoch)
    one_tune_dir = os.path.join(root_dir, project)
    if not os.path.exists(one_tune_dir):
        os.mkdir(one_tune_dir)
    model_dir = os.path.join(one_tune_dir, "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    graph_dir = os.path.join(one_tune_dir, "graphs")
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    last_graphs_dir = os.path.join(root_dir, "last_graphs")
    if not os.path.exists(last_graphs_dir):
        os.mkdir(last_graphs_dir)

    # separate valid data to valid data and test data
    if "kaika" in dp.method:
        x_valid, valid_sensors, y_valid, x_test, test_sensors, y_test = sep_kaika(dp, valid_ar, valid_dt, x_valid, valid_sensors, y_valid)
    elif "area" in dp.method:
        x_valid, valid_sensors, y_valid, x_test, test_sensors, y_test = sep_kaika(dp, valid_ar, valid_dt, x_valid, valid_sensors, y_valid, target_num = 0)
    else:
        x_test = x_valid
        y_test = y_valid

    # thin tddata
    tmp_x_train = []
    tmp_y_train = []
    tmp_train_sensors = []
    for i in range(len(x_train)):
        if i % dp.tddataThin == 0:
            tmp_x_train.append(x_train[i])
            tmp_y_train.append(y_train[i])
            tmp_train_sensors.append(train_sensors[i])
    x_train = np.array(tmp_x_train)
    y_train = np.array(tmp_y_train)
    train_sensors = np.array(tmp_train_sensors)

    # adjust data length
    x_train = x_train[0:(len(x_train) / hps.batch_size) * hps.batch_size]
    y_train = y_train[0:(len(y_train) / hps.batch_size) * hps.batch_size]
    x_valid = x_valid[0:(len(x_valid) / hps.batch_size) * hps.batch_size]
    y_valid = y_valid[0:(len(y_valid) / hps.batch_size) * hps.batch_size]
    x_test = x_test[0:(len(x_test) / hps.batch_size) * hps.batch_size]
    y_test = y_test[0:(len(x_test) / hps.batch_size) * hps.batch_size]

    if args.is_init_model:
        if args.is_sensor:
            model = CNN_4f2()
        else:
            model = CNN_42()
        pickle.dump(model, open(args.tuned_model, 'wb'))
        model.to_gpu()
    else:
        model = pickle.load(open(args.tuned_model, 'rb'))
        model.to_gpu()
    
    print "USING MODEL: %s"%model.__class__.__name__
    
    # setup optimizer
    optimizer = select_optimizer(hps.optimizer, hps.learning_rate)
    optimizer.setup(model)
    
    # train roop
    epoch = 0
    jump = x_train.shape[0] / hps.batch_size
    perm = np.random.permutation(x_train.shape[0])
    train_errors = Errors()
    valid_errors = Errors()
    min_mae = 9999
    model.get_all_weights(graph_dir, epoch)
    axes, lines = init_plot(args, hps, y_train, y_valid)
    print('hyperparameter    learning rate: {}, batch size: {}, layer num: {}, rnn units: {}, sequence length: {}, optimizer: {}, dropout: {}').format(hps.learning_rate, hps.batch_size, hps.layer_num, hps.rnn_units, hps.seq_length, hps.optimizer, hps.dropout)
    for i in six.moves.range(args.n_epoch * jump):
        # make batch
        x_batch_idx = [(jump * j + i) % x_train.shape[0] for j in xrange(hps.batch_size)]
        y_batch_idx = [(jump * j + i) % x_train.shape[0] for j in xrange(hps.batch_size)]

        x_batch = cuda.to_gpu(np.array(x_train[perm[x_batch_idx]]))
        sensors_batch = cuda.to_gpu(np.array(train_sensors[perm[x_batch_idx]]))
        y_batch = cuda.to_gpu(np.array(y_train[perm[y_batch_idx]]))
        
        # forward and calculate loss value
        if args.is_sensor:
            predict, loss_i = model.forward_one_step(hps, x_batch, sensors_batch, y_batch, train=True)
        else:
            predict, loss_i = model.forward_one_step(hps, x_batch, y_batch, train=True)

        optimizer.zero_grads()
        loss_i.backward()
        optimizer.update()

        if (i + 1) % jump == 0: 
            perm = np.random.permutation(x_train.shape[0])        
            epoch += 1
            print('epoch {}/{}').format(epoch, args.n_epoch)
           
            # evaluate train data
            py = evaluate(hps, model, x_train, train_sensors, y_train, args.is_sensor)
            train_errors.set_errors(y_train, py, None)
            # visualization 
            y = np.delete(y_train, range(py.shape[0], y_train.shape[0]), 0) # uniform y_train length to py length 
            lines.train1.set_ydata(y)
            lines.train2.set_ydata(py)
            print "+TRAIN: ",;
            axes.train.set_title(train_errors.print_now())
            axes.train.set_ylim(np.min([np.min(y),np.min(py)]), np.max([np.max(y),np.max(py)]))
            
            # evaluate validation data
            py = evaluate(hps, model, x_valid, valid_sensors, y_valid, args.is_sensor)
            valid_errors.set_errors(y_valid, py, None)
            # visualization 
            y = np.delete(y_valid, range(py.shape[0], y_valid.shape[0]), 0) # uniform y_train length to py length
            lines.valid1.set_ydata(y)
            lines.valid2.set_ydata(py)
            print "+VALID: ",;
            axes.valid.set_title(valid_errors.print_now())
            axes.valid.set_ylim(np.min([np.min(y),np.min(py)]), np.max([np.max(y),np.max(py)]))
                        
            lines.e1.set_data(range(epoch), train_errors.mae_list)
            lines.e2.set_data(range(epoch), valid_errors.mae_list)
            axes.e.set_ylim(np.min([np.min(valid_errors.mae_list),np.min(train_errors.mae_list)]), np.max([np.max(valid_errors.mae_list),np.max(train_errors.mae_list)]))
            
            if epoch == 1 or epoch % 25 == 0 or args.n_epoch == epoch:
                if abs(min_mae - valid_errors.mae_list[-1]) > 0.001 and valid_errors.mae_list[-1] - min_mae < 0.005:
                    if min_mae > valid_errors.mae_list[-1]:
                        min_mae = valid_errors.mae_list[-1]
                else:
                    plt.savefig(os.path.join(graph_dir, "rmse_" + str(epoch).zfill(4) + ".jpg"))
                    plt.savefig(os.path.join(last_graphs_dir, project + ".jpg"))
                    pickle.dump(model, open('%s/latest.chainermodel'%(model_dir), 'wb'), -1)
                    pickle.dump(optimizer, open('%s/opt.chainermodel'%(model_dir), 'wb'), -1)
                    break
                plt.savefig(os.path.join(graph_dir, "rmse_" + str(epoch).zfill(4) + ".jpg"))
                model.get_all_weights(graph_dir, epoch)
                if args.n_epoch == epoch:
                    plt.savefig(os.path.join(last_graphs_dir, project + ".jpg"))
                    pickle.dump(model, open('%s/latest.chainermodel'%(model_dir), 'wb'), -1)
                    pickle.dump(optimizer, open('%s/opt.chainermodel'%(model_dir), 'wb'), -1)
                    
            plt.pause(.01)
            model.reset_state()
            
            # decay learning rate depending on epoch
            if epoch >= args.learning_rate_decay_after:
                if hps.optimizer == 'SGD' or hps.optimizer == 'MomentumSGD' or hps.optimizer == 'NesterovAG':
                    optimizer.lr *= args.learning_rate_decay
                else:
                    pass
    '''
    # evaluate test data
    test_errors = Errors()
    py = evaluate(hps, model, x_test, test_sensors, y_test, args.is_sensor)
    test_errors.set_errors(y_test, py, None)
       
    # visualize test result
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1, 1, 1)
    print "+TEST: ", 
    plt.title(test_errors.print_now())
    y = np.delete(y_test, range(py.shape[0], y_test.shape[0]), 0) # uniform y_train length to py length
    vis_len_test = (len(y) / hps.batch_size) * hps.batch_size
    ax.plot(range(vis_len_test), y, label="true")
    ax.plot(range(vis_len_test), py, "orange", alpha=0.7, label="pred")
    plt.savefig(os.path.join(graph_dir, "test.jpg"))

    plt.close('all')

    # save hyperparameters and rmse to csv
    with open(errors_csv, 'a+') as f:
        writer = csv.writer(f)
        body = vars(hps).values() + train_errors.get_last_errors() + valid_errors.get_last_errors() + test_errors.get_last_errors()
        writer.writerow(body)
    '''
         
if __name__=='__main__':
    # initialize params    
    args = init_args()

    if not os.path.exists("/".join(args.tuned_model.split("/")[:-1])):
        os.makedirs("/".join(args.tuned_model.split("/")[:-1]))

    
    # load train data, validation data, test data
    dp = init_data_params()
    pname, x_train, y_train, x_valid, y_valid, train_sensors, valid_sensors, train_ar, train_dt, valid_ar, valid_dt = load_data(dp)

    # make directory for each tuning
    root_dir = os.path.join(args.out_path, pname)
    if not os.path.exists(root_dir):
        #os.mkdir(root_dir)
        os.makedirs(root_dir)
    
    # write parameter at this tune
    with open(os.path.join(root_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(str(k) + ':' + str(v) + '\n')
            
    n_tune = 100
    for i in range(n_tune):
        print('Tune Count: {}/{}').format(i, n_tune)
        hps = sampling_hyperparameters()
        main(dp, hps, args, root_dir, x_train, y_train, x_valid, y_valid, train_sensors, valid_sensors, valid_ar, valid_dt)

