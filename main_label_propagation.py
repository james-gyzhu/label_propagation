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
print '\n----- This running is to propagate labels based on labeled/unlabeled samples -----\n'


# import packages
from time import time
import numpy as np
from sklearn.semi_supervised import LabelPropagation

import glb_utils


# load raw data
load_param = dict(data_type='sklearn_builtin_data', data_name='iris')
t_start = time()
raw_data = glb_utils.load_data(param_dict=load_param)
t_end = time()
print 'Data loading taking %f seconds' % (t_end - t_start)
print '\n'


# do label propagation
label_prop_model = LabelPropagation()

random_unlabeled_points = np.where(np.random.random_integers(0, 1, size=len(raw_data.target)))
labels = np.copy(raw_data.target)
labels[random_unlabeled_points] = -1
prop_model = label_prop_model.fit(raw_data.data, labels)


# all done
print '----- All Done -----\n'

i = 0