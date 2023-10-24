from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GroupShuffleSplit
from skopt import BayesSearchCV
from src.base_model import BaseModel
from src.classification.model.base_classifier_model import BaseClassfierModel

DEFAULT_PARAMS = {
    # 'hidden_layer_sizes': [(50, 50)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': (0.0001, 0.05, 'log-uniform'),
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.01, 0.05, 0.1, 0.5],
    'max_iter': [100, 200, 300],
}

class MLPNetClassifier(BaseClassfierModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator = MLPClassifier()

    @property
    def default_params(self):
        return DEFAULT_PARAMS
