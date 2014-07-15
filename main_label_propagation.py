#!/usr/bin/env python

__author__ = 'Guangyu Zhu'
__copyright__ = 'Copyright (C) 2014, Solutions for Label Propagation'
__license__ = 'Domain Upon Request'
__maintainer__ = 'James Zhu'
__email__ = 'zhugy.cn@gmail.com'

"""

Main file to propagate labels based on labeled/unlabeled samples

"""


# output intro message
print('\n----- This running is to propagate labels based on labeled/unlabeled samples -----\n')


# import packages
from time import time
import numpy as np

from sklearn import datasets
import scikit_algo


# parsing arguments
unlabeled_sample_selection = False
num_sample_used = 330
num_labeled_points = 10


# load raw data
t_start = time()
digits = datasets.load_digits()
t_end = time()
print('Data loading taking %f seconds' % (t_end - t_start))
print('Data loaded: %d/%d with labels %d' % (digits.data.shape[0], digits.data.shape[1], len(digits.target)))
print('\n')


# select labeled samples (form index list)
if unlabeled_sample_selection:
    rng = np.random.RandomState(0)
    indices = np.arange(len(digits.data))
    rng.shuffle(indices)

    sample_idx_fn = 'labeled_sample_list.idx'
    import pickle
    pkl_file = open(sample_idx_fn, 'wb')
    pickle.dump(obj=indices, file=pkl_file, protocol=-2)
    pkl_file.close()
    print('Dump labeled sample list to: %s\n' % sample_idx_fn)
else:
    sample_idx_fn = 'labeled_sample_list.idx'
    import pickle
    pkl_file = open(sample_idx_fn, 'rb')
    indices = pickle.load(pkl_file)
    pkl_file.close()
    print('Load labeled sample list from: %s\n' % sample_idx_fn)


# form semi-supervised data set
print('Form semi-supervised data for modeling')
num_sample_used = len(digits.target) if num_sample_used > len(digits.target) else num_sample_used
num_labeled_points = num_sample_used if num_labeled_points > num_sample_used else num_labeled_points
print('Number of samples to form data: %d' % num_sample_used)
print('Number of labeled samples in data: %d' % num_labeled_points)

data_x = digits.data[indices[:num_sample_used]]
data_y = digits.target[indices[:num_sample_used]]
images = digits.images[indices[:num_sample_used]]

num_total_samples = len(data_y)
unlabeled_indices = np.arange(num_total_samples)[num_labeled_points:]

data_y_train = np.copy(data_y)
data_y_train[unlabeled_indices] = -1
print('\n')


# build label propagation model
lp_param = dict(prop_algo='LabelSpreading', iter_num=5, gamma=0.25, max_iter=5, trace=True)
lp_model_inst = scikit_algo.LabelPropScikit(data_x=data_x, data_y_train=data_y_train, data_y_true=data_y,
                                            param_dict=lp_param)
lp_model = lp_model_inst.iter_label_prop


# all done
print('----- All Done -----\n')

i = 0