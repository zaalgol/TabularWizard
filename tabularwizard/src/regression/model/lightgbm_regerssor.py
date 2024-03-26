#https://www.kaggle.com/code/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from src.regression.base_regressor_model import BaseRegressorrModel

DEFAULT_PARAMS = {
    'learning_rate': (0.01, 0.3, 'log-uniform'),  # typical range from learning rate
    'num_leaves': (31, 200),  # depends on max_depth, should be smaller than 2^(max_depth)
    'max_depth': (3, 11),  # typical values can range from 3-10
    'min_child_samples': (10, 200),  # minimum number of data needed in a child (leaf)
    'min_child_weight': (1e-5, 1e-3, 'log-uniform'),  # deals with under-fitting
    'subsample': (0.5, 1.0, 'uniform'),  # commonly used range
    'subsample_freq': (1, 10),  # frequency for bagging
    'colsample_bytree': (0.5, 1.0, 'uniform'),  # fraction of features that can be selected for each tree
    'reg_alpha': (1e-9, 10.0, 'log-uniform'),  # L1 regularization term
    'reg_lambda': (1e-9, 10.0, 'log-uniform'),  # L2 regularization term
    'n_estimators': (50, 1000),  # number of boosted trees to fit
}

class LightGBMRegressor(BaseRegressorrModel):
    def __init__(self, train_df, prediction_column, *args, split_column=None, test_size=0.3, **kwargs):
        super().__init__(train_df, prediction_column, split_column=split_column, test_size=test_size)
        self.estimator = LGBMRegressor(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
  