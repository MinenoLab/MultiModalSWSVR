# coding: utf-8

from collections import namedtuple
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import NullLocator
from chainer import cuda
import os

def init_plot(args, hps, y_train, y_valid):
    # common setting
    fig, (ax_train, ax_valid, ax_e) = plt.subplots(3, 1, figsize=(20,10))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0.9)
    
    # set graph title
    ax_train.set_title("Training data Prediction: RMSE on %d epoch -> %f" % (0, 0))
    ax_valid.set_title("Validation data Prediction: RMSE on %d epoch -> %f" % (0, 0))
    ax_e.set_title("RMSE every epoch")
    
    # set axis
    #ax_train.set_ylim(0, 1)
    #ax_valid.set_ylim(0, 1)
    #ax_e.set_ylim(0, 1)
    
    # plot initialized value
    vis_len_train = (len(y_train) / hps.batch_size) * hps.batch_size
    vis_len_valid = (len(y_valid) / hps.batch_size) * hps.batch_size
    lines_train1, = ax_train.plot(range(vis_len_train), y_train[:vis_len_train], label="true")
    lines_train2, = ax_train.plot(range(vis_len_train), y_train[:vis_len_train], "orange", alpha=0.6, label="pred", markersize=3)
    lines_valid1, = ax_valid.plot(range(vis_len_valid), y_valid[:vis_len_valid], label="true")
    lines_valid2, = ax_valid.plot(range(vis_len_valid), y_valid[:vis_len_valid], "green", alpha=0.6, label="pred", markersize=3)    
    lines_e1, = ax_e.plot(range(args.n_epoch), range(args.n_epoch), "orange", label="train")
    lines_e2, = ax_e.plot(range(args.n_epoch), range(args.n_epoch), "green", label="test")
    
    AxSetting = namedtuple('AxSetting', 'train valid e')
    axes = AxSetting(ax_train, ax_valid, ax_e)
    LineSetting = namedtuple('LineSetting', 'train1 train2 valid1 valid2 e1 e2')
    lines = LineSetting(lines_train1, lines_train2, lines_valid1, lines_valid2, lines_e1, lines_e2)
    return axes, lines

def showPlot(layer, name):
    W = layer.params().next()
    fig = plt.figure()
    fig.patch.set_facecolor('black')
    fig.suptitle(W.label, fontweight="bold")
    plot(W)
    fig.savefig(name)
    plt.close(fig)
    
def plot(W):
    float32=0
    dim = eval('('+W.label+')')[0]
    size = int(math.ceil(math.sqrt(dim[0])))
    if(len(dim)==4):
        for i,channel in enumerate(W.data):
            new_channel = cuda.to_cpu(channel).copy()
            ax = plt.subplot(size,size, i+1)
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
            accum = new_channel[0]
            for ch in new_channel:
                accum += ch
            accum /= len(new_channel)
            ax.imshow(accum, interpolation='nearest')
    else:
        plt.imshow(W.data, interpolation='nearest')