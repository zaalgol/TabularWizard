from sklearn.neighbors import KNeighborsClassifier
from src.classification.model.base_classifier_model import BaseClassfierModel


DEFAULT_PARAMS = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': list(range(20, 51)),
    'p': [1, 2]
}

class KnnClassifier(BaseClassfierModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimator = KNeighborsClassifier()

    @property
    def default_params(self):
        return DEFAULT_PARAMS