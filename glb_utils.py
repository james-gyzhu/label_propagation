__author__ = 'Guangyu Zhu'
__copyright__ = 'Copyright (C) 2014, Solutions for Label Propagation'
__license__ = 'Domain Upon Request'
__maintainer__ = 'James Zhu'
__email__ = 'zhugy.cn@gmail.com'

"""

This file defines all global utility functions

"""


# import packages
import pandas as pd
import pandas.rpy.common as com
import scipy.sparse as sp_sparse
import numpy as np

from sklearn import datasets


# global macro
load_config_defaults = {
    'data_type': 'sklearn_builtin_data',
    'data_name': 'iris'
}

read_config_defaults = {
    'sep': '\|',
    'header': 0,
    'drop_duplicates': False,
    'index_attr': 'LYL_CRD_NO'
}

write_config_defaults = {
    'sep': '|',
    'header': True,
    'index': False,
    'cols': None
}


# define functions

def load_data(param_dict=load_config_defaults):
    load_param = load_config_defaults.copy()
    load_param.update(param_dict)

    data_df = None
    if 'sklearn_builtin_data' == param_dict['data_type']:
        if 'iris' == param_dict['data_name']:
            data_df = datasets.load_iris()
        else:
            print 'No specific data set in scikit-learn package\n'
    elif 'file_data' == param_dict['data_type']:
        if not param_dict['data_name']:
            data_df = read_table_to_frame(data_fn=param_dict['data_name'])
        else:
            print 'Error in file name to load data\n'

    return data_df


def read_table_to_frame(data_fn, param_dict=read_config_defaults):
    # check input parameters
    if not data_fn:
        print 'Error in file name to load data\n'
        return None

    read_param = read_config_defaults.copy()
    read_param.update(param_dict)

    # load data given file name
    try:
        raw_data = pd.read_table(data_fn, header=read_param['header'], sep=read_param['sep'])
    except IOError:
        print 'Unable to open the data file\n'
        return None

    # trim head/tail whitespace of column name
    raw_data.columns = map(str.strip, raw_data.columns)

    # remove duplicates if needs
    if read_param['drop_duplicates']:
        raw_data = raw_data.reset_index().groupby(raw_data[param_dict['index_attr']]).first()

    # sort rows by index
    raw_data = raw_data.sort(read_param['index_attr'], axis=0, ascending=True)

    # return data in DataFrame structure
    return raw_data


def write_frame_to_table(df, fn, param_dict=write_config_defaults):
    if (df is None) or (not fn):
        print 'Error in file name to write data\n'
        return -1

    write_param = write_config_defaults.copy()
    write_param.update(param_dict)

    # write data to file
    try:
        df.to_csv(fn, sep=write_param['sep'], header=write_param['header'],
                  index=write_param['index'], cols=write_param['cols'])
    except IOError:
        print 'Unable to write the data file\n'
        return -1

    return 1


def convert_pframe_to_rframe(pframe):
    # check input parameters
    if (pframe is None) or (not isinstance(pframe, pd.DataFrame)):
        print 'Error in pandas object data frame\n'
        return None

    # convert to R object data frame
    rframe = com.convert_to_r_dataframe(df=pframe, strings_as_factors=True)

    # return data
    return rframe


def is_sparse(x):
    sparse_flag = sp_sparse.issparse(x)
    return sparse_flag


def scale_data(x, center, scale):
    scaled_data = x
    if center:
        col_sum = x.sum(axis=2)
        scaled_data = scaled_data - col_sum[:, np.newaxis]

    if scale:
        col_std = x.std(axis=2)
        scaled_data = scaled_data / col_std[:, np.newaxis]

    return scaled_data