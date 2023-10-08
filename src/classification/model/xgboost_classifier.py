import matplotlib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
# from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

EVAL_METRIC = 'rmsle'
TEST_SIZE = 0.25
N_ITER = 25
SCORING = 'neg_mean_squared_error'


class XgboostClassifier:
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

    def tune_hyper_parameters_with_bayesian (self):
        params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                  'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                  'subsample': np.arange (0.5, 1.0, 0.1),
                  'colsample_bytree': np.arange (0.4, 1.0, 0.1),
                  'colsample_bylevel': np.arange (0.4, 1.0, 0.1),
                  'min_child_weight': (0, 30),
                  'gamma': [0, 1, 5],
                  'n_estimators': (50, 1000),
                  'reg_alpha': [0, 0.1, 0.2, 0.3, 1],
                  'reg_lambda': [1, 1.5, 1.6, 1.7, 1.8, 2]}
        xgbr = xgb.XGBClassifier (enable_categorical=True, tree_method='hist')
        self.search = BayesSearchCV (estimator=xgbr,
                                     search_spaces=params,
                                     scoring=SCORING,
                                     n_iter=N_ITER,
                                     verbose=0)

    def train (self):
        eval_set = [(self.X_test, self.y_test)]
        print(self.X_test)
        result = self.search.fit (self.X_train, self.y_train, early_stopping_rounds=10, eval_metric= "mlogloss",
                                  eval_set=eval_set)  # warning msg
        print ("Best parameters:", self.search.best_params_)
        print ("Lowest RMSE: ", (-self.search.best_score_) ** (1 / 2.0))
        return result

    def plot (self, result):
        from xgboost import plot_importance
        import matplotlib.pyplot as plt
        plt.style.use ('fivethirtyeight')
        plt.rcParams.update ({'font.size': 16})

        fig, ax = plt.subplots (figsize=(12, 6))
        plot_importance (result.best_estimator_, max_num_features=8, ax=ax)
        plt.show()

        # xgb.plot_tree (result.best_estimator_, num_trees=2)
        # fig = matplotlib.pyplot.gcf ()
        # fig.set_size_inches (150, 100)
        # fig.savefig ('tree.png')

        # plot_tree (result.best_estimator_)

    def predict_test (self):
        self.test_predictions = self.search.predict (self.X_test)

    def predict_train (self):
        self.train_predictions = self.search.predict (self.X_train)

    def predict_true (self):
        self.true_predictions = self.search.predict (self.X_true)

    # def predict (self):
    #     self.test_predictions = self.search.predict (self.true_data)

    def evaluate_predictions (self):
        cm = confusion_matrix (self.y_test, self.test_predictions)
        print (classification_report (self.y_test, self.test_predictions))
        print (f"confusion_matrix of test is {cm}")

        cm = confusion_matrix (self.y_train, self.train_predictions)
        print (classification_report (self.y_train, self.train_predictions))
        print (f"confusion_matrix of train is {cm}")

        #
        # RMSLE = np.sqrt (mean_squared_log_error (self.y_test, self.predictions))
        # print ("The score is %.5f" % RMSLE)
        # return RMSLE;