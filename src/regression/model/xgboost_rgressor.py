import matplotlib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from xgboost import plot_tree
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import KFold

eval_metric = 'rmsle'
TEST_SIZE = 0.25

class XgboostRegressor:
    def __init__ (self, train_df,
                prediction_column,
                split_column=None):
        self.X_true = None
        self.train_predictions = None
        self.test_predictions = None
        self.search = None,

        if split_column is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split (train_df,
                                                                                        train_df [prediction_column],
                                                                                        test_size=TEST_SIZE)
        else:
            splitter = GroupShuffleSplit (test_size=TEST_SIZE, n_splits=2, random_state=7)
            split = splitter.split (train_df, groups=train_df [split_column])
            train_inds, test_inds = next (split)

            train = train_df.iloc [train_inds]
            # self.X_train = train.drop ([prediction_column], axis=1)
            self.y_train = train [[prediction_column]].astype(float)
            test = train_df.iloc [test_inds]
            #self.X_test = test.drop ([prediction_column], axis=1)
            self.y_test = test [[prediction_column]].astype(float)


        self.X_train = self.X_train.drop ([prediction_column], axis=1)
        self.X_test = self.X_test.drop ([prediction_column], axis=1)
        # self.X_test = self.X_test.select_dtypes (include=['number']).copy ()

    def tune_hyper_parameters_with_bayesian(self, params_constrained=None):
        params = {
            # 'max_depth': (3, 11, 1),
            # 'learning_rate': (0.01, 1.0, "log-uniform"),
            # 'subsample': (0.01, 1.0, "uniform"),
            # "gamma": (1e-9, 0.5, "log-uniform"),
            # 'colsample_bytree': np.arange(0.4, 1.0, 0.1),
            # 'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
            # 'n_estimators': (50, 1000)
            
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

        xgbr = xgb.XGBRegressor(enable_categorical=True, tree_method='hist',
                                early_stopping_rounds=10, eval_metric="rmse",
                                interaction_constraints=params_constrained
                                )
        kfold = KFold(n_splits=10)
        self.search = BayesSearchCV(estimator=xgbr,
                                       search_spaces=params,
                                       scoring='neg_mean_squared_error',
                                       n_iter=25,
                                       cv=kfold,
                                       verbose=0)

    def train (self):
        eval_set = [(self.X_test, self.y_test)]
        print(self.X_test)
        result = self.search.fit(self.X_train, self.y_train,
                                  eval_set=eval_set)  # warning msg
        print ("Best parameters:", self.search.best_params_)
        print ("Lowest RMSE: ", (-self.search.best_score_) ** (1 / 2.0))
        return result

    def plot(self, result):
        plot_importance(result.best_estimator_)
        plt.show()

        # from xgboost import plot_importance
        # plt.style.use ('fivethirtyeight')
        # plt.rcParams.update ({'font.size': 16})
        #
        # fig, ax = plt.subplots (figsize=(12, 6))
        # plot_importance(result.best_estimator_, max_num_features=8, ax=ax)
        # plt.show ();

        # xgb.plot_tree (result.best_estimator_, num_trees=2)
        # fig = matplotlib.pyplot.gcf ()
        # fig.set_size_inches (150, 100)
        # fig.savefig ('tree.png')

        # plot_tree (result.best_estimator_)

    # def predict(self, model, x):
    #     return model.predict(x)
    #
    # def compare_with_true_data(self, true_data, predictions):
    #     from sklearn.metrics import mean_squared_log_error
    #     RMSLE = np.sqrt(mean_squared_log_error(true_data, predictions))
    #     RMSE = np.sqrt(((predictions - true_data) ** 2).mean())
    #     ABSOLUTE = np.absolute(true_data - predictions).mean()
    #     print("The RMSLE is %.5f" % RMSLE)
    #     print("The RMSE is %.5f" % RMSE)
    #     print("The ABSOLUTE is %.5f" % ABSOLUTE)
    #     return RMSE, RMSLE, ABSOLUTE;