from catboost import CatBoostClassifier
from src.classification.model.base_classifier_model import BaseClassfierModel


DEFAULT_PARAMS = {
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'iterations': [100, 500, 1000]

}

class CatboostClassifier(BaseClassfierModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator = CatBoostClassifier(task_type = 'GPU', devices='0')

    @property
    def default_params(self):
        return DEFAULT_PARAMS