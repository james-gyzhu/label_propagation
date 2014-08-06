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
                            dlp_T=20, alpha=0.05, beta=0.1, trace=True)


class DynamicLabelPropagation(object):
    def __init__(self, data_x, data_y_train, data_y_true=None, param_dict=prop_config_defaults):
        # labeled sample index
        labeled_sample_index = np.where(data_y_train != -1)[0]
        unlabeled_sample_index = np.where(data_y_train == -1)[0]

        # re-array labeled & unlabeled samples
        self.num_labeled_sample = len(labeled_sample_index)
        self.data_x = np.concatenate((data_x[labeled_sample_index, :], data_x[unlabeled_sample_index, :]), axis=0)
        self.data_y_train = np.concatenate((data_y_train[labeled_sample_index], data_y_train[unlabeled_sample_index]))
        self.data_y_true = np.concatenate((data_y_true[labeled_sample_index], data_y_true[unlabeled_sample_index]))

        # set parameters
        self.prop_config = prop_config_defaults.copy()
        self.prop_config.update(param_dict)
        print('Label propagation configuration is:')
        pprint.pprint(self.prop_config)
        print('\n')

    def dynamic_label_prop(self):
        alpha = self.prop_config['alpha']
        beta = self.prop_config['beta']

        # calculate transition matrix
        transition_mat = self.calc_transition_mat()

        # calculate status matrix
        status_mat = self.calc_status_mat(transition_mat)

        # calculate knn matrix
        knn_mat = self.calc_knn_graph(transition_mat)

        # calculate label matrix
        unique_label_vec = [np.unique(self.data_y_train) != -1]  # remove -1 (indicator for unlabeled samples)
        labeled_y_mat = np.zeros(shape=(self.num_labeled_sample, len(unique_label_vec)))
        labeled_y_mat[range(labeled_y_mat.shape[0]), self.data_y_train[:self.num_labeled_sample]] = 1
        unlabeled_y_mat = np.zeros(shape=(len(self.data_y_train) - self.num_labeled_sample, len(unique_label_vec)))

        # iterative calculation
        label_mat_t = np.concatenate((labeled_y_mat, unlabeled_y_mat), axis=0)
        status_mat_t = status_mat

        for t in range(self.prop_config['dlp_T']):
            label_mat_tp1 = np.dot(status_mat_t, label_mat_t)
            label_mat_tp1[range(labeled_y_mat.shape[0]), range(labeled_y_mat.shape[1])] = labeled_y_mat

            status_mat_tp1 = np.dot(
                np.dot(knn_mat, status_mat_t + alpha * np.dot(label_mat_t, np.transpose(label_mat_t))),
                np.transpose(knn_mat)) + beta * np.identity(status_mat_t.shape[0])

            label_mat_t = label_mat_tp1
            status_mat_t = status_mat_tp1

        return label_mat_t

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
            transition_mat = np.exp(- distance_mat ** 2)
        else:
            print('Error in kernel metric for transition matrix calculation\n')
            return None

        return transition_mat

    def calc_status_mat(self, raw_mat):
        row_sum = np.sum(raw_mat, axis=1)
        status_mat = raw_mat / row_sum[:, np.newaxis]
        return status_mat

    def calc_knn_graph(self, raw_mat):
        sort_idx_mat = np.argsort(raw_mat, axis=1)[:, raw_mat.shape[1]-self.prop_config['n_neighbors']:]
        knn_graph_mat = np.zeros(shape=raw_mat.shape)
        for i_row in range(knn_graph_mat.shape[0]):
            knn_graph_mat[i_row, sort_idx_mat[i_row, :]] = raw_mat[i_row, sort_idx_mat[i_row, :]]

        return knn_graph_mat


def iter_label_propagation(data_x, data_y_train, data_y_true, param_dict=prop_config_defaults):
    lp_model_inst = DynamicLabelPropagation(data_x=data_x, data_y_train=data_y_train, data_y_true=data_y_true,
                                            param_dict=param_dict)
    iter_lp_model = lp_model_inst.dynamic_label_prop()

    return iter_lp_model

