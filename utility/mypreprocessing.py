# coding: utf-8

import numpy as np

def global_contrast_normalize(self, X, scale=1., subtract_mean=True, 
    use_std=False, sqrt_bias=0., min_divisor=1e-8):

    ndim = X.ndim
    if not ndim in [3,4]: raise NotImplementedError("X.dim>4 or X.ndim<3")

    scale = float(scale)
    mean = X.mean(axis=ndim-1)
    new_X = X.copy()

    if subtract_mean:
        if ndim==3:
            new_X = X - mean[:,:,None]
        else: new_X = X - mean[:,:,:,None]

    if use_std:
        normalizers = T.sqrt(sqrt_bias + X.var(axis=ndim-1)) / scale
    else:
        normalizers = T.sqrt(sqrt_bias + (new_X ** 2).sum(axis=ndim-1)) / scale

    # Don't normalize by anything too small.
    T.set_subtensor(normalizers[(normalizers < min_divisor).nonzero()], 1.)

    if ndim==3: new_X /= normalizers[:,:,None]
    else: new_X /= normalizers[:,:,:,None]

    return new_X
