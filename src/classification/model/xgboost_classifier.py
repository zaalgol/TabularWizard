from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GroupShuffleSplit
from skopt import BayesSearchCV

DEFAULT_PARAMS = {
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'n_estimators': list(range(50, 300, 50)),
    'max_depth': list(range(3, 10)),
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': list(range(1, 10)),
    'gamma': [i/10.0 for i in range(0, 5)],
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100],
    'scale_pos_weight': [1, 10, 25, 50, 75, 99, 100, 1000]
}

class XgboostClassifier:
    def __init__(self, train_df, prediction_column, split_column=None, test_size=0.3):
        self.search = None

        if split_column is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                train_df, train_df[prediction_column], test_size=test_size, random_state=7)
        else:
            splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=7)
            split = splitter.split(train_df, groups=train_df[split_column])
            train_inds, test_inds = next(split)

            train = train_df.iloc[train_inds]
            self.y_train = train[[prediction_column]].astype(float)
            test = train_df.iloc[test_inds]
            self.y_test = test[[prediction_column]].astype(float)

        self.X_train = self.X_train.drop([prediction_column], axis=1)
        self.X_test = self.X_test.drop([prediction_column], axis=1)

    def tune_hyper_parameters(self, params=DEFAULT_PARAMS, scoring='neg_mean_squared_error', kfold=10, n_iter=25):
        Kfold = KFold(n_splits=kfold)  
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss') 
        self.search = BayesSearchCV(estimator=xgb,
                                    search_spaces=params,
                                    scoring=scoring,
                                    n_iter=n_iter,
                                    cv=Kfold,
                                    verbose=0)

    def train(self):
        result = self.search.fit(self.X_train, self.y_train)
        print("Best parameters:", self.search.best_params_)
        print("Best accuracy:", self.search.best_score_)

        return result


    # def plot (self, result):
    #     from xgboost import plot_importance
    #     import matplotlib.pyplot as plt
    #     plt.style.use ('fivethirtyeight')
    #     plt.rcParams.update ({'font.size': 16})

    #     fig, ax = plt.subplots (figsize=(12, 6))
    #     plot_importance (result.best_estimator_, max_num_features=8, ax=ax)
    #     plt.show()

    #     # xgb.plot_tree (result.best_estimator_, num_trees=2)
    #     # fig = matplotlib.pyplot.gcf ()
    #     # fig.set_size_inches (150, 100)
    #     # fig.savefig ('tree.png')

    #     # plot_tree (result.best_estimator_)

    # def predict_test (self):
    #     self.test_predictions = self.search.predict (self.X_test)

    # def predict_train (self):
    #     self.train_predictions = self.search.predict (self.X_train)

    # def predict_true (self):
    #     self.true_predictions = self.search.predict (self.X_true)

    # def evaluate_predictions (self):
    #     cm = confusion_matrix (self.y_test, self.test_predictions)
    #     print (classification_report (self.y_test, self.test_predictions))
    #     print (f"confusion_matrix of test is {cm}")

    #     cm = confusion_matrix (self.y_train, self.train_predictions)
    #     print (classification_report (self.y_train, self.train_predictions))
    #     print (f"confusion_matrix of train is {cm}")
