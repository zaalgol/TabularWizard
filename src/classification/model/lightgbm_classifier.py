from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GroupShuffleSplit
from skopt import BayesSearchCV
from src.base_model import BaseModel
from src.classification.model.base_classifier_model import BaseClassfierModel


DEFAULT_PARAMS = {
            'class_weight': [None, 'balanced'],
            'boosting_type': ['gbdt', 'goss', 'dart'],
            'num_leaves': list(range(30, 150)),
            'learning_rate': [0.01,0.05, 0.1 ,0.5],
            'subsample_for_bin': [20000,50000,100000,120000,150000],
            'min_child_samples': [20,50,100,200,500],
            'colsample_bytree': [0.6,0.8,1],
            "max_depth": (5,100, 'uniform'),
            'lambda_l1': (0.7, 5, 'log-uniform'),
            'lambda_l2': (0.7, 5, 'log-uniform')
        }

class LightgbmClassifier(BaseClassfierModel):
    def __init__(self, a=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator = LGBMClassifier()
        self.default_params = DEFAULT_PARAMS

    @property
    def default_params(self):
        return DEFAULT_PARAMS

