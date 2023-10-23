from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GroupShuffleSplit
from skopt import BayesSearchCV

DEFAULT_PARAMS_RF = {
    'n_estimators': list(range(50, 300, 50)),
    'max_depth': list(range(3, 10)) + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'max_features': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
}

class RandomForestClassifierCustom:
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

    def tune_hyper_parameters(self, params=DEFAULT_PARAMS_RF, scoring='neg_mean_squared_error', kfold=10, n_iter=25):
        Kfold = KFold(n_splits=kfold)  
        rf = RandomForestClassifier() 
        self.search = BayesSearchCV(estimator=rf,
                                    search_spaces=params,
                                    scoring=scoring,
                                    n_iter=n_iter,
                                    cv=Kfold,
                                    verbose=0)

    def train(self):
        result = self.search.fit(self.X_train, self.y_train.values.ravel())  # using values.ravel() to get a 1-D array
        print("Best parameters:", self.search.best_params_)
        print("Best accuracy:", self.search.best_score_)

        return result
