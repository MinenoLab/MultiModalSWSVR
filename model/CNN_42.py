# coding: utf-8

import numpy as np
import chainer
from chainer import cuda, Variable
import chainer.functions as F
import chainer.links as L
import os
from chainer.functions.array import concat, reshape
from visualize.myplot import showPlot


class CNN_42(chainer.Chain):
    def __init__(self, train=True):
        initializer = chainer.initializers.HeNormal()
        super(CNN_42, self).__init__(
            conv1_r=L.Convolution2D(3, 24, 3, stride=2, pad=1, initialW=initializer),  # for rgb image
            conv1_m=L.Convolution2D(3, 24, 3, stride=2, pad=1, initialW=initializer),  # for pool mask
            prelu1_r=L.PReLU((24, 72, 72)),
            prelu1_m=L.PReLU((24, 72, 72)),
            bn1_r=L.BatchNormalization(24),
            bn1_m=L.BatchNormalization(24),

            conv2_r=L.Convolution2D(24, 48, 3, stride=1, pad=1, initialW=initializer),  # for rgb image
            conv2_m=L.Convolution2D(24, 48, 3, stride=1, pad=1, initialW=initializer),  # for pool mask
            prelu2_r=L.PReLU((48, 36, 36)),
            prelu2_m=L.PReLU((48, 36, 36)),
            bn2_r=L.BatchNormalization(48),
            bn2_m=L.BatchNormalization(48),

            conv3_r=L.Convolution2D(48, 96, 3, stride=2, pad=1, initialW=initializer),  # for rgb image
            conv3_m=L.Convolution2D(48, 96, 3, stride=2, pad=1, initialW=initializer),  # for pool mask
            prelu3_r=L.PReLU((96, 9, 9)),
            prelu3_m=L.PReLU((96, 9, 9)),
            bn3_r=L.BatchNormalization(96),
            bn3_m=L.BatchNormalization(96),

            cccp4=L.Convolution2D(96 * 2, 96, 1, initialW=initializer),
            bn4=L.BatchNormalization(96),

            conv5=L.Convolution2D(96, 96, 3, stride=2, pad=1, initialW=initializer),
            prelu5=L.PReLU((96, 3, 3)),
            bn5=L.BatchNormalization(96),

            fc6=L.Linear(6, 384, initialW=initializer),
            prelu6=L.PReLU((384)),
            bn6=L.BatchNormalization(384),

            cccp7=L.Convolution2D(2, 1, 1, initialW=initializer),
            
            fc8=L.Linear(384, 64, initialW=initializer),
            prelu8=L.PReLU((64)),
            bn8=L.BatchNormalization(64),
            
            fc9=L.Linear(64, 1, initialW=initializer),
        )
    
    def forward_one_step(self, hps, x_data, y_data, train=True):
        x_r = Variable(x_data[:, 0:3, :, :], volatile=not train)
        t = Variable(y_data, volatile=not train)

        cn1_r = F.max_pooling_2d(self.prelu1_r(self.bn1_r(self.conv1_r(x_r), test=not train)), ksize=2, stride=2, pad=0)

        cn2_r = F.max_pooling_2d(self.prelu2_r(self.bn2_r(self.conv2_r(cn1_r), test=not train)), ksize=2, stride=2, pad=0)

        cn3_r = F.max_pooling_2d(self.prelu3_r(self.bn3_r(self.conv3_r(cn2_r), test=not train)), ksize=2, stride=2, pad=0)

        cn5_rm = F.max_pooling_2d(self.prelu5(self.bn5(self.conv5(cn3_r), test=not train)), ksize=2, stride=2, pad=0)
        
        cs8 = F.dropout(self.prelu8(self.bn8(self.fc8(cn5_rm), test=not train)), ratio=hps.dropout, train=train)
        
        y = self.fc9(cs8)

        return y, F.mean_squared_error(y, t)
    
    def get_weights(self, hps, x_data, train=False):
        x_r = Variable(x_data[:, 0:3, :, :], volatile=not train)

        cn1_r = F.max_pooling_2d(self.prelu1_r(self.bn1_r(self.conv1_r(x_r), test=not train)), ksize=2, stride=2, pad=0)
        cn2_r = F.max_pooling_2d(self.prelu2_r(self.bn2_r(self.conv2_r(cn1_r), test=not train)), ksize=2, stride=2, pad=0)
        cn3_r = F.max_pooling_2d(self.prelu3_r(self.bn3_r(self.conv3_r(cn2_r), test=not train)), ksize=2, stride=2, pad=0)
        cn5_rm = F.max_pooling_2d(self.prelu5(self.bn5(self.conv5(cn3_r), test=not train)), ksize=2, stride=2, pad=0)
        cs8 = F.dropout(self.prelu8(self.bn8(self.fc8(cn5_rm), test=not train)), ratio=hps.dropout, train=train)
        
        return cuda.to_cpu(cs8.data)

    def reset_state(self):
        pass

    def get_all_weights(self, graph_dir, epoch):
        for attr in dir(self):
            if not "conv" in attr: continue
            showPlot(getattr(self, attr), os.path.join(graph_dir, attr + "_" + str(epoch) + ".jpg"))