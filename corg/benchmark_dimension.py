""" CORG """

from sklearn.base import BaseEstimator, TransformerMixin

import os.path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut

import numpy as np

class BenchmarkDimension(BaseEstimator, TransformerMixin): 

    def __init__(self, compute_train_error = True, random_state = None):

        self.random_state = random_state

        # if True return train error, otherwise perform CV
        self.compute_train_error = compute_train_error

    def fit(self, X, Y): 

        if not isinstance(X, pd.DataFrame):
            raise ValueError('\'X\' parameter must be a pandas dataframe')

        if 'entity' not in X.columns:
            raise ValueError('\'X\' has to have an \'entity\' column')

        if len(X.columns) != 2:
            raise ValueError('\'X\' has to have two columns')

        dimension_name = X.columns.tolist()[0]
        if dimension_name == 'entity':
            dimension_name = X.columns.tolist()[1]

        if not isinstance(Y, pd.DataFrame):
            raise ValueError('\'Y\' parameter must be a pandas dataframe') 

        if 'entity' not in Y.columns:
            raise ValueError('\'Y\' has to have an \'entity\' column')

        if 'label' not in Y.columns:
            raise ValueError('\'Y\' has to have an \'label\' column')

        XY = pd.merge(X, Y, on = 'entity', how = 'inner')

        if (len(XY.label.unique())) != 2: # labels should be binary
                raise ValueError('Labels should be binary')
        
        X_np = XY[dimension_name].values.reshape(-1, 1)
        y_np = XY['label'].values

        clf_model = LogisticRegression(random_state = self.random_state)
        clf_model.fit(X_np, y_np)

        # compute logistic regression coefficients
        self.beta1_ = clf_model.coef_[0].tolist()[0]
        self.beta0_ = clf_model.intercept_[0]

        if self.compute_train_error:
            y_train_pred_np = clf_model.predict(X_np)

            self.accuracy_train_ = accuracy_score(y_np, y_train_pred_np)
            self.precision_train_ = precision_score(y_np, y_train_pred_np)
            self.recall_train_ = recall_score(y_np, y_train_pred_np)
            self.f1_score_train_ = f1_score(y_np, y_train_pred_np)
        else:
            cv = LeaveOneOut()

            # define scores to compute
            scoring = {'accuracy' : make_scorer(accuracy_score),
                    'precision' : make_scorer(precision_score, average = 'micro'),
                    'recall' : make_scorer(recall_score, average = 'micro'),
                    'f1_score' : make_scorer(f1_score, average = 'micro')}
            scores = cross_validate(clf_model, X_np, y_np, scoring = scoring, cv = cv, n_jobs = -1)

            self.accuracy_mean_ = np.mean(scores['test_accuracy'])
            self.accuracy_std_ = np.std(scores['test_accuracy'])

            self.precision_mean_ = np.mean(scores['test_precision'])
            self.precision_std_ = np.std(scores['test_precision'])

            self.recall_mean_ = np.mean(scores['test_recall'])
            self.recall_std_ = np.std(scores['test_recall'])

            self.f1_score_mean_ = np.mean(scores['test_f1_score'])
            self.f1_score_std_ = np.std(scores['test_f1_score'])

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

    def load_dimension_from_file(self, path_dimension_file, dimension,
            dimension_file_header_names = None):

        # check that a dimension file is provided
        if path_dimension_file is None:
            raise ValueError('Dimensions file name is not provided.')

        # check that a dimension file exists
        if not os.path.isfile(path_dimension_file):
            raise ValueError('Dimensions file does not.')

        # handles files with or without header
        header_df = pd.read_csv(path_dimension_file, nrows = 0)
        column_no = len(header_df.columns)
        if column_no < 2:
            raise ValueError('Dimensions file has to have at least two columns.') 

        # sanity checks in header
        if dimension_file_header_names is not None:
            if dimension_file_header_names['entity'] not in header_df.columns:
                raise ValueError('Dimensions file has to have a ' 
                        + dimension_file_header_names['entity'] + ' column.') 

        # load dimensions data
        dim_df = None
        if dimension_file_header_names is None:
            dim_df = pd.read_csv(path_dimension_file, header = None).rename(columns = {0:'entity'})
        else:
            dim_df = pd.read_csv(path_dimension_file).rename(columns 
                    = {dimension_file_header_names['entity']:'entity'}) 

        if dimension is None:
            raise ValueError('A dimension to benchmark needs to be provided.')

        if dimension not in dim_df.columns:
            raise ValueError('A dimension to benchmark does not exist.')

        dim_df = dim_df[['entity', dimension]]

        dim_df.dropna(inplace = True)

        dim_df['entity'] = dim_df['entity'].astype(str)
        dim_df[dimension] = dim_df[dimension].astype(float)

        return(dim_df)

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

        # sanity checks in header
        if label_file_header_names is not None:
            if label_file_header_names['entity'] not in header_df.columns:
                raise ValueError('Dimensions file has to have a ' 
                        + label_file_header_names['entity'] + ' column.') 

        # load dimensions data
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
        label_df['label'] = label_df['label'].astype(str)

        return(label_df)
