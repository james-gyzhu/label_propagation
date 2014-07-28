__author__ = 'Guangyu Zhu'
__copyright__ = 'Copyright (C) 2014, Solutions for Label Propagation'
__license__ = 'Domain Upon Request'
__maintainer__ = 'James Zhu'
__email__ = 'zhugy.cn@gmail.com'

"""

Dynamic Label propagation

"""


# import packages
import pprint
from scipy.spatial.distance import *

import glb_utils


# global macro
prop_config_defaults = dict(dist_metric='euclidean', kernel='rbf', gamma=20, n_neighbors=7,
                            alpha=0.05, beta=0.1, trace=True)


class DynamicLabelPropagation(object):
    def __init__(self, data_x, data_y_train, data_y_true=None, param_dict=prop_config_defaults):
        self.data_x, self.data_y_train, self.data_y_true = data_x, data_y_train, data_y_true

        self.prop_config = prop_config_defaults.copy()
        self.prop_config.update(param_dict)
        print('Label propagation configuration is:')
        pprint.pprint(self.prop_config)
        print('\n')

    def dynamic_label_prop(self):
        dlp_model = None

        # labeled sample index
        labeled_sample_index = np.where(self.data_y_train != -1)[0]

        # calculate label matrix
        unique_label_vec = np.unique(self.data_y_train)
        unique_label_vec = unique_label_vec[unique_label_vec != -1]  # remove -1 (indicator for unlabeled samples)
        label_mat = np.zeros(shape=(len(self.data_y_train), len(unique_label_vec)))
        for i in labeled_sample_index:
            label_mat[i, np.where(unique_label_vec == self.data_y_train[i])[0]] = self.data_y_train[i]

        # calculate transition matrix
        transition_mat = self.calc_transition_mat()
        if not transition_mat:
            return None

        # calculate status matrix
        status_mat = self.calc_status_mat(transition_mat)

        # calculate knn matrix
        knn_mat = self.calc_knn_graph(transition_mat)

        return dlp_model

    def calc_transition_mat(self):
        # center/scale data
        scaled_x = glb_utils.scale_data(x=self.data_x, center=True, scale=True)

        # calculate distance matrix
        if 'euclidean' == self.prop_config['dist_metric']:
            distance_mat = cdist(XA=scaled_x, XB=scaled_x, metric='euclidean')
        else:
            print('Error in distance metric for transition matrix calculation\n')
            return None

        # calculate transaction matrix
        if 'rbf' == self.prop_config['kernel']:
            transition_mat = np.exp(-distance_mat**2)
        else:
            print('Error in kernel metric for transition matrix calculation\n')
            return None

        return transition_mat

    def calc_status_mat(self, raw_mat):
        row_sum = raw_mat.sum(axis=1)
        status_mat = raw_mat / row_sum[:, np.newaxis]
        return status_mat

    def calc_knn_graph(self, raw_mat):
        sort_idx_mat = np.argsort(raw_mat, axis=1)[:, :self.prop_config['n_neighbors']]
        knn_graph_mat = np.zeros(shape=(raw_mat.shape[0], raw_mat.shape[1]))
        knn_graph_mat[sort_idx_mat] = raw_mat[sort_idx_mat]
        return knn_graph_mat


def iter_label_propagation(data_x, data_y_train, data_y_true, param_dict=prop_config_defaults):
    lp_model_inst = DynamicLabelPropagation(data_x=data_x, data_y_train=data_y_train, data_y_true=data_y_true,
                                            param_dict=param_dict)
    iter_lp_model = lp_model_inst.dynamic_label_prop()

    return iter_lp_model