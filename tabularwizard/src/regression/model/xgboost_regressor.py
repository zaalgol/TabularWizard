import matplotlib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from xgboost import XGBRegressor, plot_tree
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import KFold
from tabularwizard.src.regression.base_regressor_model import BaseRegressorrModel


DEFAULT_PARAMS = {
            'max_depth': (3, 10, 1),
            'learning_rate': (0.01, 0.3, "log-uniform"),
            'subsample': (0.5, 1.0, "uniform"),
            "gamma": (1e-9, 0.5, "log-uniform"),
            'colsample_bytree': (0.5, 1.0, "uniform"),
            'colsample_bylevel': (0.5, 1.0, "uniform"),
            'n_estimators': (100, 1000),
            'alpha': (0, 1),
            'lambda': (0, 1),
            'min_child_weight': (1, 10)
        }

class XgboostRegressor(BaseRegressorrModel):
    def __init__(self, train_df, prediction_column, *args, split_column=None, test_size=0.3, **kwargs):
        super().__init__(train_df, prediction_column, split_column=split_column, test_size=test_size)
        self.estimator = XGBRegressor(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
    
    def plot(self, result):
        plot_importance(result.best_estimator_)
        plt.show()

# class XgboostRegressor(BaseModel):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def tune_hyper_parameters(self, params_constrained=None, hyperparams=None,
#                                              tree_method = "hist",device = None,  early_stopping_rounds=10, eval_metric='rmse',
#                                              scoring='neg_mean_squared_error', n_iter = 25, verbose = 0):
#         if hyperparams is None:
#             self.hyperparams = DEFAULT_PARAMS
#         else:
#             self.hyperparams = hyperparams

#         xgbr = xgb.XGBRegressor(enable_categorical=True, tree_method = tree_method, device = device,
#                                 early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric,
#                                 interaction_constraints=params_constrained
#                                 )
#         kfold = KFold(n_splits=10)
#         self.search = BayesSearchCV(estimator=xgbr,
#                                        search_spaces=self.hyperparams,
#                                        scoring=scoring,
#                                        n_iter=n_iter,
#                                        cv=kfold,
#                                        verbose=verbose)
        

#     def train (self):
#         eval_set = [(self.X_test, self.y_test)]
#         print(self.X_test)
#         result = self.search.fit(self.X_train, self.y_train,
#                                   eval_set=eval_set)  
#         print ("Best parameters:", self.search.best_params_)
#         print ("Lowest RMSE: ", (-self.search.best_score_) ** (1 / 2.0))
#         return result

