from __future__ import division
import numpy as np
import sys

sys.path.append("./mingqingscript")

import scipy.io as sio
import scipy.ndimage.interpolation
# import scipy.signal

import os

import math
import random

import pdb
import random
import numpy as np
import pickle
import random
import sys
import shutil

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

# torch condiguration
import argparse
from math import log10
# import scipy.io as sio
import numpy as np

import random
from random import uniform
import h5py
import time
import PIL
from PIL import Image

import h5py
import numpy as np
import matplotlib.pyplot as plt

# plt.axis([0, 10, 0, 1])
plt.ion()


# for i in range(10):
#     y = np.random.random()
#     plt.scatter(i, y)
#     plt.pause(0.05)
#
# while True:
#     plt.pause(0.05)
def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255 * numpy.ones((len(arr), 1), numpy.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


index = 1
nyu_depth = h5py.File('nyu_depth_v2_labeled.mat', 'r')

directory='facades/train'

if not os.path.exists(directory):
    os.makedirs(directory)


image = nyu_depth['images']
depth = nyu_depth['depths']

img_size = 224

# per=np.random.permutation(1400)
# np.save('rand_per.py',per)
# pdb.set_trace()
total_num = 0
plt.ion()
for index in range(1000):
    index = index
    gt_image = (image[index, :, :, :]).astype(float)
    gt_image = np.swapaxes(gt_image, 0, 2)

    gt_image = scipy.misc.imresize(gt_image, [img_size, img_size]).astype(float)

    gt_image = gt_image / 255


    gt_depth = depth[index, :, :]
    maxhazy = gt_depth.max()
    minhazy = gt_depth.min()
    gt_depth = (gt_depth) / (maxhazy)

    gt_depth = np.swapaxes(gt_depth, 0, 1)
    scale1 = (gt_depth.shape[0]) / img_size
    scale2 = (gt_depth.shape[1]) / img_size

    gt_depth = scipy.ndimage.zoom(gt_depth, (1 / scale1, 1 / scale2), order=1)

    if gt_depth.shape != (img_size, img_size):
        continue

    for j in range(8):

        beta = uniform(0.5, 2)

        tx1 = np.exp(-beta * gt_depth)

        a = 1 - 0.5 * uniform(0, 1)


        m = gt_image.shape[0]
        n = gt_image.shape[1]

        rep_atmosphere = np.tile(np.reshape(A, [1, 1, 3]), [m, n, 1])
        tx1 = np.reshape(tx1, [m, n, 1])

        max_transmission = np.tile(tx1, [1, 1, 3])

        haze_image = gt_image * max_transmission + rep_atmosphere * (1 - max_transmission)

        total_num = total_num + 1
        scipy.misc.imsave('a0.9beta1.29.jpg', haze_image)
        scipy.misc.imsave('gt.jpg', gt_image)

        h5f=h5py.File('./facades/train/'+str(total_num)+'.h5','w')
        h5f.create_dataset('haze',data=haze_image)
        h5f.create_dataset('trans',data=max_transmission)
        h5f.create_dataset('atom',data=rep_atmosphere)
        h5f.create_dataset('gt',data=gt_image)
