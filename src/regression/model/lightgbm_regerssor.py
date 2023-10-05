#https://www.kaggle.com/code/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

EVAL_METRIC = 'rmse'
TEST_SIZE = 0.3
DEFAULT_HYPER_PARAMETERS = {
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


class LightGBMRegressor:
    def __init__(self, train_df, prediction_column, test_df=None):
        self.X_true = None
        self.train_predictions = None
        self.test_predictions = None
        self.search = None
        self.hyperparams = None

        if test_df is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                train_df, train_df[prediction_column], test_size=TEST_SIZE
            )
        else:
            self.X_train = train_df 
            self.y_train = train_df[prediction_column]
            self.X_test = test_df
            self.y_test = test_df[prediction_column]

        self.X_train = self.X_train.drop([prediction_column], axis=1)
        self.X_test = self.X_test.drop([prediction_column], axis=1)

    def tune_hyper_parameters_with_bayesian(self, params_constrained=None, hyperparams=None,
                                            early_stopping_rounds=10, eval_metric=EVAL_METRIC,
                                            scoring='neg_mean_squared_error', n_iter=25, verbose=0):
        if hyperparams is None:
            self.hyperparams = DEFAULT_HYPER_PARAMETERS
        else:
            self.hyperparams = hyperparams

        lgbr = lgb.LGBMRegressor(metric=eval_metric, early_stopping_round=early_stopping_rounds,
                                num_iterations=10000, n_jobs=-1, categorical_feature=params_constrained)
        
        kfold = KFold(n_splits=10)
        self.search = BayesSearchCV(estimator=lgbr,
                                    search_spaces=self.hyperparams,
                                    scoring=scoring,
                                    n_iter=n_iter,
                                    cv=kfold,
                                    verbose=verbose)

    def train(self):
        eval_set = [(self.X_test, self.y_test)]
        result = self.search.fit(self.X_train, self.y_train, eval_set=eval_set, eval_metric=EVAL_METRIC)
        print("Best parameters:", self.search.best_params_)
        print("Lowest RMSE: ", (-self.search.best_score_) ** (1 / 2.0))
        return result
