from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GroupShuffleSplit
from skopt import BayesSearchCV

DEFAULT_PARAMS = {
            'class_weight': [None, 'balanced'],
            'boosting_type': ['gbdt', 'goss', 'dart'],
            'num_leaves': list(range(30, 150)),
            'learning_rate': [0.01,0.05, 0.1 ,0.5],
            'subsample_for_bin': [20000,50000,100000,120000,150000],
            'min_child_samples': [20,50,100,200,500],
            'colsample_bytree': [0.6,0.8,1],
            # "max_depth": [5,10,50,100],
            "max_depth": (5,100, 'uniform'),
            # 'lambda_l1': [0, 0.1, 0.5, 1],
            # 'lambda_l2': [0, 0.1, 0.5, 1]
            'lambda_l1': (0.7, 1),
            'lambda_l2': (0.7, 1)
        }

class LightgbmClassifier:
    def __init__(self, train_df, prediction_column, split_column=None, test_size = 0.3):
        self.X_true = None
        self.train_predictions = None
        self.test_predictions = None
        self.search = None
        self.test_size = test_size

        if split_column is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_df,
                                                                                     train_df[prediction_column],
                                                                                     test_size=self.test_size)
        else:
            splitter = GroupShuffleSplit(test_size=self.test_size, n_splits=2, random_state=7)
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
        lgbm = LGBMClassifier()
        self.search = BayesSearchCV(estimator=lgbm,
                                    search_spaces=params,
                                    scoring=scoring,
                                    n_iter=n_iter, 
                                    cv=Kfold,
                                    verbose=0)

    def train(self):
        result = self.search.fit(self.X_train, self.y_train)
        print("Best parameters:", self.search.best_params_)
        print("Best accuracy: ", self.search.best_score_)

        y_train_pred = result.predict(self.X_train) 
        score = round(accuracy_score(self.y_train, y_train_pred), 3) 
        print("train acuracy", score)

        y_test_pred = result.predict(self.X_test)
        score = round(accuracy_score(self.y_test, y_test_pred), 3)
        print("test acuracy", score)

        return result, self.search.best_params_, self.search.best_score_, score
