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
from sklearn.metrics import classification_report, confusion_matrix

import glb_utils


# global macro
prop_config_defaults = dict(iter_num=1,  # iteration number for iterative label propagation
                            dist_metric='euclidean', kernel='rbf', gamma=20, n_neighbors=7,
                            dlp_T=20, alpha=0.05, beta=0.1, trace=True)


class DynamicLabelPropagation(object):
    def __init__(self, data_x, data_y_train, param_dict=prop_config_defaults):
        # labeled sample index
        labeled_sample_index = np.where(data_y_train != -1)[0]
        unlabeled_sample_index = np.where(data_y_train == -1)[0]
        self.orig_sample_index = np.concatenate((labeled_sample_index, unlabeled_sample_index))

        # re-array labeled & unlabeled samples
        self.num_labeled_sample = len(labeled_sample_index)
        self.data_x = np.concatenate((data_x[labeled_sample_index, :], data_x[unlabeled_sample_index, :]), axis=0)
        self.data_y_train = np.concatenate((data_y_train[labeled_sample_index], data_y_train[unlabeled_sample_index]))

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
        status_mat = self.calc_status_mat(raw_mat=transition_mat)

        # calculate knn matrix
        knn_mat = self.calc_knn_graph(raw_mat=transition_mat)
        knn_mat = self.calc_status_mat(raw_mat=knn_mat)

        # calculate label matrix
        class_vec = np.unique(self.data_y_train)
        class_vec = (class_vec[class_vec != -1])  # remove -1 (indicator for unlabeled samples)

        unlabeled_y_mat = np.zeros(shape=(len(self.data_y_train) - self.num_labeled_sample, len(class_vec)))
        labeled_y_mat = np.zeros(shape=(self.num_labeled_sample, len(class_vec)))
        for i_row in range(labeled_y_mat.shape[0]):
            labeled_y_mat[i_row, np.where(class_vec == self.data_y_train[i_row])[0][0]] = 1

        # iterative calculation
        label_mat_t = np.concatenate((labeled_y_mat, unlabeled_y_mat), axis=0)
        status_mat_t = status_mat

        for t in range(self.prop_config['dlp_T']):
            # label propagation
            label_mat_tp1 = np.dot(status_mat_t, label_mat_t)
            # label clamping
            label_mat_tp1[range(labeled_y_mat.shape[0]), :] = labeled_y_mat
            # kernel fusion/diffusion
            status_mat_tp1 = np.dot(
                np.dot(knn_mat, status_mat_t + alpha * np.dot(label_mat_t, np.transpose(label_mat_t))),
                np.transpose(knn_mat)) + beta * np.identity(status_mat_t.shape[0])

            # get back propagation matrix for next iteration
            label_mat_t = label_mat_tp1
            status_mat_t = status_mat_tp1

        transducted_y = np.concatenate((self.data_y_train[:self.num_labeled_sample],
                                        class_vec[np.argmax(label_mat_t[self.num_labeled_sample:, ], axis=1)]))

        # format output
        dlp_model = FormatModel(dlp_parm=self.prop_config, orig_sample_index=self.orig_sample_index,
                                labeled_index=range(0, self.num_labeled_sample),
                                unlabeled_index=range(self.num_labeled_sample, self.data_x.shape[0]),
                                classes=class_vec, transduction=transducted_y)

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
        sort_idx_mat = np.argsort(-raw_mat, axis=1)[:, :self.prop_config['n_neighbors']]
        knn_graph_mat = np.zeros(shape=raw_mat.shape)
        for i_row in range(knn_graph_mat.shape[0]):
            knn_graph_mat[i_row, sort_idx_mat[i_row, :]] = raw_mat[i_row, sort_idx_mat[i_row, :]]

        return knn_graph_mat


class FormatModel(object):
    def __init__(self, dlp_parm, orig_sample_index, labeled_index, unlabeled_index, classes, transduction):
        self.dlp_param_ = dlp_parm
        self.orig_sample_index_ = orig_sample_index
        self.labeled_index_ = labeled_index
        self.unlabeled_index_ = unlabeled_index
        self.classes_ = classes
        self.transduction_ = transduction


def iter_label_propagation(data_x, data_y_train, data_y_true=None, param_dict=prop_config_defaults):
    iter_lp_model = None
    for i in range(param_dict['iter_num']):
        lp_model_inst = DynamicLabelPropagation(data_x=data_x, data_y_train=data_y_train, param_dict=param_dict)
        iter_lp_model = lp_model_inst.dynamic_label_prop()

        if not (data_y_true is None):
            data_y_true = data_y_true[iter_lp_model.orig_sample_index_]

            # performance report via confusion matrix
            predicted_labels = iter_lp_model.transduction_[iter_lp_model.unlabeled_index_]
            true_labels = data_y_true[iter_lp_model.unlabeled_index_]
            cm = confusion_matrix(true_labels, predicted_labels, labels=iter_lp_model.classes_)

            print(classification_report(true_labels, predicted_labels))
            print('Confusion matrix')
            print(cm)
            print('\n')

        if iter_lp_model.dlp_param_['trace']:
            print('Iteration %i %s' % (i, 60 * '_'))
            print('Label propagation model: %d labeled & %d unlabeled (%d total)'
                  % (len(iter_lp_model.labeled_index_), len(iter_lp_model.unlabeled_index_),
                     len(iter_lp_model.transduction_)))

    return iter_lp_model

