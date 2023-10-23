from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GroupShuffleSplit
from skopt import BayesSearchCV

DEFAULT_PARAMS = {
    # 'hidden_layer_sizes': [(50, 50)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': (0.0001, 0.05, 'log-uniform'),
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.01, 0.05, 0.1, 0.5],
    'max_iter': [100, 200, 300],
}

class MLPNetClassifier:
    def __init__(self, train_df, prediction_column, split_column=None, test_size=0.3):
        self.search = None

        if split_column is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_df,
                                                                                    train_df[prediction_column],
                                                                                    test_size=test_size, random_state=7)
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

    def tune_hyper_parameters(self, params=DEFAULT_PARAMS, scoring='neg_mean_squared_error', kfold=10, n_iter=50):
        Kfold = KFold(n_splits=kfold)  
        mlp = MLPClassifier()
        self.search = BayesSearchCV(estimator=mlp,
                                    search_spaces=params,
                                    scoring=scoring,
                                    n_iter=n_iter, 
                                    cv=Kfold,
                                    verbose=0)

    def train(self):
        result = self.search.fit(self.X_train, self.y_train)
        print("Best parameters:", self.search.best_params_)
        print("Best accuracy: ", self.search.best_score_)

        return result
