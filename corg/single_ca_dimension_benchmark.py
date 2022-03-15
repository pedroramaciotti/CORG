""" CORD """

from sklearn.base import BaseEstimator, TransformerMixin

import os.path

import pandas as pd

#import numpy as np

class SingleCADimensionBenchmark(BaseEstimator, TransformerMixin): 

    def __init__(self, compute_train_error = True, random_state = None):

        self.random_state = random_state

        # if True return train error, otherwise perform CV
        self.compute_train_error = compute_train_error

    def fit(self, X, Y):
        return self

    def transform(self, X):
        return (X)

    def get_params(self, deep = True):
        return {'random_state': self.random_state, 
                'compute_train_error': self.compute_train_error}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

    def score(self, X, y):
        return 1

    def load_CA_feature_from_file(self, path_ca_dimension_file, ca_dimension,
            ca_dimension_file_header_names = None):

        # check that a CA dimension file is provided
        if path_ca_dimension_file is None:
            raise ValueError('CA dimensions file name is not provided.')

        # check that CA dimension file exists
        if not os.path.isfile(path_ca_dimension_file):
            raise ValueError('CA dimensions file does not.')

        # handles files with or without header
        header_df = pd.read_csv(path_ca_dimension_file, nrows = 0)
        column_no = len(header_df.columns)
        if column_no < 2:
            raise ValueError('CA dimensions file has to have at least two columns.') 

        # sanity checks in header
        if ca_dimension_file_header_names is not None:
            if ca_dimension_file_header_names['entity'] not in header_df.columns:
                raise ValueError('CA dimensions file has to have a ' 
                        + ca_dimension_file_header_names['entity'] + ' column.') 

        # load ca dimensions data
        ca_dim_df = None
        if ca_dimension_file_header_names is None:
            ca_dim_df = pd.read_csv(path_ca_dimension_file, header = None).rename(columns = {0:'entity'})
        else:
            ca_dim_df = pd.read_csv(path_ca_dimension_file).rename(columns 
                    = {ca_dimension_file_header_names['entity']:'entity'}) 

        if ca_dimension is None:
            raise ValueError('A CA dimension to benchmark needs to be provided.')

        if ca_dimension not in ca_dim_df.columns:
            raise ValueError('A CA dimension to benchmark does not exist.')

        ca_dim_df = ca_dim_df[['entity', ca_dimension]]

        ca_dim_df.dropna(inplace = True)

        ca_dim_df['entity'] = ca_dim_df['entity'].astype(str)
        ca_dim_df[ca_dimension] = ca_dim_df[ca_dimension].astype(float)

        return(ca_dim_df)

    def load_label_from_file(self, path_label_file, label_file_header_names = None):

        # check that a label filename is provided
        if path_label_file is None:
            raise ValueError('Label file name is not provided.')

        # check that label file exists
        if not os.path.isfile(path_label_file):
            raise ValueError('Label file does not.')
        
        # handles files with or without header
        header_df = pd.read_csv(path_label_file, nrows = 0)
        column_no = len(header_df.columns)
        if column_no < 2:
            raise ValueError('Label file has to have at least two columns.') 

        '''
        # sanity checks in header
        if ca_dimension_file_header_names is not None:
            if ca_dimension_file_header_names['entity'] not in header_df.columns:
                raise ValueError('CA dimensions file has to have a ' 
                        + ca_dimension_file_header_names['entity'] + ' column.') 
        '''

        # load ca dimensions data
        label_df = None
        if label_file_header_names is None:
            label_df = pd.read_csv(path_label_file, header = None).rename(columns 
                    = {0:'entity', 1:'label'})
        else:
            label_df = pd.read_csv(path_label_file).rename(columns 
                    = {label_file_header_names['entity']:'entity', 
                        label_file_header_names['label']:'label'})

        label_df = label_df[['entity', 'label']]

        label_df.dropna(inplace = True)

        label_df['entity'] = label_df['entity'].astype(str)
        label_df['label'] = label_df['label'].astype(float)

        return(label_df)
