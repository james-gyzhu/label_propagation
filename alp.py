__author__ = 'Guangyu Zhu'
__copyright__ = 'Copyright (C) 2014, Solutions for Label Propagation'
__license__ = 'Domain Upon Request'
__maintainer__ = 'James Zhu'
__email__ = 'zhugy.cn@gmail.com'

"""

Active label propagation (ALP)

"""


# import packages
import numpy as np
from scipy import stats
import pprint
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix


# global macro
prop_config_defaults = dict(prop_algo='LabelSpreading',  # LabelSpreading/LabelPropagation
                            active_iter=1,  # iteration number for active propagation
                            kernel='rbf', gamma=20, n_neighbors=7, alpha=0.2, max_iter=30, tol=1e-3,
                            n_init_samples=10, n_active_samples=5, trace=True)


class ActiveLabelPropagation(object):
    def __init__(self, param_dict):
        # update parameter configure
        prop_config = prop_config_defaults.copy()
        prop_config.update(param_dict)
        print('Label propagation configuration is:')
        pprint.pprint(prop_config)
        print('\n')

        # active learning parameters
        self.prop_algo = prop_config['prop_algo']
        self.active_iter = prop_config['active_iter']
        self.n_init_samples = prop_config['n_init_samples']
        self.n_active_samples = prop_config['n_active_samples']

        # propagation parameters
        self.max_iter = prop_config['max_iter']
        self.tol = prop_config['tol']

        # kernel parameters
        self.kernel = prop_config['kernel']
        self.gamma = prop_config['gamma']
        self.n_neighbors = prop_config['n_neighbors']

        # clamping factor
        self.alpha = prop_config['alpha']

        # info output
        self.trace = prop_config['trace']

    def fit(self, data_x, data_y_train, data_y_true=None):
        if data_x.shape[0] != len(data_y_train):
            print('Error on input data in propagation\n')
            return None
        if (not (data_y_true is None)) and (len(data_y_train) != len(data_y_true)):
            print('Error on input data in propagation\n')
            return None

        # get indices of labeled/unlabeled samples
        labeled_indices = np.where(data_y_train != -1)[0]
        unlabeled_indices = np.where(data_y_train == -1)[0]  # unlabeled samples labeling with -1
        if len(labeled_indices) < self.n_init_samples:
            print('Error: %d labeled samples not enough for initial at %d'
                  % (len(labeled_indices), self.n_init_samples))
            return None
        else:
            used_labeled_indices = labeled_indices[:self.n_init_samples]

        active_lp_model, running_iter = None, 1
        while (running_iter <= self.active_iter) and (len(used_labeled_indices) <= len(labeled_indices)):
            # set labels at current iteration
            cur_iter_y = -np.ones(len(data_y_train))  # -1 as initialization
            cur_iter_y[used_labeled_indices] = data_y_train[used_labeled_indices]

            # print tracing info
            print('Iteration %i %s' % (running_iter, 60*'_'))
            print('Label propagation model: %d labeled & %d unlabeled (%d total)'
                  % (len(used_labeled_indices), len(unlabeled_indices), len(data_y_train)))

            # initialize propagation model instant
            if 'LabelSpreading' == self.prop_algo:
                active_lp_model = label_propagation.LabelSpreading(gamma=self.gamma, max_iter=self.max_iter)
            else:
                active_lp_model = label_propagation.LabelPropagation(gamma=self.gamma, max_iter=self.max_iter)

            # fit propagation model
            active_lp_model.fit(data_x, cur_iter_y)

            # performance report via confusion matrix
            if self.trace and (not (data_y_true is None)):
                predicted_labels = active_lp_model.transduction_[unlabeled_indices]
                true_labels = data_y_true[unlabeled_indices]
                cm = confusion_matrix(true_labels, predicted_labels, labels=active_lp_model.classes_)
                print(classification_report(true_labels, predicted_labels))
                print('Confusion matrix')
                print(cm)

            # compute the entropies of transduced label distributions
            pred_entropies = stats.distributions.entropy(active_lp_model.label_distributions_.T)

            # select five digit examples that the classifier is most uncertain about
            unused_labeled_indices = list(set(labeled_indices) - set(used_labeled_indices))
            n_selected_samples = min(len(unused_labeled_indices), self.n_active_samples)
            uncertainty_index = np.argsort(pred_entropies)[unused_labeled_indices][-n_selected_samples:]

            # keep track of indices that we get labels for
            for index in uncertainty_index:
                add_index = np.where(labeled_indices == index)[0]
                used_labeled_indices = np.concatenate((used_labeled_indices, add_index))

            running_iter += 1
            print('\n')

        return active_lp_model


def alp_label_propagation(data_x, data_y_train, data_y_true=None, param_dict=prop_config_defaults):
    alp_model_inst = ActiveLabelPropagation(param_dict=param_dict)
    alp_model = alp_model_inst.fit(data_x=data_x, data_y_train=data_y_train, data_y_true=data_y_true)
    return alp_model