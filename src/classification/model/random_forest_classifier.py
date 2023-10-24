from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GroupShuffleSplit
from skopt import BayesSearchCV
from src.base_model import BaseModel
from src.classification.model.base_classifier_model import BaseClassfierModel

DEFAULT_PARAMS = {
    'n_estimators': list(range(50, 300, 50)),
    'max_depth': list(range(3, 10)) + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'max_features': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
}

class RandomForestClassifierCustom(BaseClassfierModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator = RandomForestClassifier()

    @property
    def default_params(self):
        return DEFAULT_PARAMS

    def train(self):
        result = self.search.fit(self.X_train, self.y_train.values.ravel())  # using values.ravel() to get a 1-D array
        print("Best parameters:", self.search.best_params_)
        print("Best accuracy:", self.search.best_score_)

        return result
