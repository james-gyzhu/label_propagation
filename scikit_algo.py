__author__ = 'Guangyu Zhu'
__copyright__ = 'Copyright (C) 2014, Solutions for Label Propagation'
__license__ = 'Domain Upon Request'
__maintainer__ = 'James Zhu'
__email__ = 'zhugy.cn@gmail.com'

"""

Label propagation using algorithms in Scikit-Learn package

"""


# import packages
import numpy as np
from scipy import stats
import pprint
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix


# global macro
prop_config_defaults = dict(prop_algo='LabelSpreading',  # LabelSpreading/LabelPropagation
                            iter_num=5,  # iteration number for iterative label propagation
                            kernel='rbf', gamma=20, n_neighbors=7, alpha=0.2, max_iter=30, tol=1e-3,
                            trace=True)


class LabelPropScikit(object):
    def __init__(self, data_x, data_y_train, data_y_true, param_dict=prop_config_defaults):
        self.data_x, self.data_y_train, self.data_y_true = data_x, data_y_train, data_y_true
        if (self.data_x.shape[0] != len(data_y_train)) or (len(data_y_train) != len(data_y_true)):
            print('Dimension Error in label propagation\n')
            return

        self.prop_config = prop_config_defaults.copy()
        self.prop_config.update(param_dict)
        print('Label propagation configuration is:')
        pprint.pprint(self.prop_config)

    @property
    def iter_label_prop(self):
        if self.prop_config['iter_num'] < 1:
            print('Iteration Error in iterative label propagation\n')
            return

        # get indices of unlabeled samples
        unlabeled_indices = np.where(self.data_y_train == -1)  # unlabeled samples labeling with -1

        # do iterative label propagation
        for i in range(self.prop_config['iter_num']):
            # initialize propagation model instant
            if 'LabelSpreading' == self.prop_config['prop_algo']:
                iter_lp_model = label_propagation.LabelSpreading(gamma=self.prop_config['gamma'],
                                                                 max_iter=self.prop_config['max_iter'])
            else:
                iter_lp_model = label_propagation.LabelPropagation(gamma=self.prop_config['gamma'],
                                                                   max_iter=self.prop_config['max_iter'])

            # fit propagation model
            iter_lp_model.fit(self.data_x, self.data_y_train)

            # performance report via confusion matrix
            predicted_labels = iter_lp_model.transduction_[unlabeled_indices]
            true_labels = self.data_y_true[unlabeled_indices]
            cm = confusion_matrix(true_labels, predicted_labels, labels=iter_lp_model.classes_)

            if self.prop_config['trace']:
                print('Iteration %i %s' % (i, 50 * '_'))
                print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
                      % (len(self.data_y_train) - len(unlabeled_indices),
                      len(unlabeled_indices), len(self.data_y_train)))

                print(classification_report(true_labels, predicted_labels))
                print("Confusion matrix")
                print(cm)

            # compute the entropies of transduced label distributions
            pred_entropies = stats.distributions.entropy(iter_lp_model.label_distributions_.T)

            # select five digit examples that the classifier is most uncertain about
            uncertainty_index = np.argsort(pred_entropies)[-5:]

            # keep track of indices that we get labels for
            delete_indices = np.array([])
            for index in uncertainty_index:
                delete_index = np.where(unlabeled_indices == index)
                delete_indices = np.concatenate((delete_indices, delete_index))

            unlabeled_indices = np.delete(unlabeled_indices, delete_indices)

        return iter_lp_model


def iter_label_propagation(data_x, data_y, param_dict=prop_config_defaults):
    lp_model = LabelPropScikit(data_x=data_x, data_y=data_y, param_dict=param_dict)
    iter_lp_model = lp_model.iter_label_prop()

    return iter_lp_model

