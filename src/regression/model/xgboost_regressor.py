import matplotlib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from xgboost import plot_tree
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import KFold

EVAL_METRIC = 'rmse'
TEST_SIZE = 0.3
DEFAULT_HYPER_PARAMETERS = {
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

class XgboostRegressor:
    def __init__ (self, train_df,
                prediction_column,
                test_df=None):
        self.X_true = None
        self.train_predictions = None
        self.test_predictions = None
        self.search = None,
        self.hyperparams = None,

        if test_df is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split (train_df,
                                                                                    train_df[prediction_column],
                                                                                    test_size=TEST_SIZE)
            
        else:
            self.X_train = train_df 
            self.y_train = train_df[prediction_column]
            self.y_test = test_df
            self.y_test = test_df [prediction_column]

        self.X_train = self.X_train.drop ([prediction_column], axis=1)
        self.X_test = self.X_test.drop ([prediction_column], axis=1)

    def tune_hyper_parameters_with_bayesian(self, params_constrained=None, hyperparams=None,
                                             tree_method = "hist",device = None,  early_stopping_rounds=10, eval_metric=EVAL_METRIC,
                                             scoring='neg_mean_squared_error', n_iter = 25, verbose = 0):
        if hyperparams is None:
            self.hyperparams = DEFAULT_HYPER_PARAMETERS
        else:
            self.hyperparams = hyperparams

        xgbr = xgb.XGBRegressor(enable_categorical=True, tree_method = tree_method, device = device,
                                early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric,
                                interaction_constraints=params_constrained
                                )
        kfold = KFold(n_splits=10)
        self.search = BayesSearchCV(estimator=xgbr,
                                       search_spaces=self.hyperparams,
                                       scoring=scoring,
                                       n_iter=n_iter,
                                       cv=kfold,
                                       verbose=verbose)
        

    def train (self):
        eval_set = [(self.X_test, self.y_test)]
        print(self.X_test)
        result = self.search.fit(self.X_train, self.y_train,
                                  eval_set=eval_set)  
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