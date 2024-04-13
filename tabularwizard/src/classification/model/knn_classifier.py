from sklearn.neighbors import KNeighborsClassifier
from tabularwizard.src.classification.model.base_classifier_model import BaseClassfierModel
from sklearn.preprocessing import StandardScaler


DEFAULT_PARAMS = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': list(range(20, 51)),
    # 'p': [1, 2]
}

class KnnClassifier(BaseClassfierModel):
    def __init__(self, train_df, prediction_column, *args, split_column=None, test_size=0.3, **kwargs):
        super().__init__(train_df, prediction_column, split_column=split_column, test_size=test_size)
        self.unique_classes = train_df[prediction_column].nunique()
        self.check_and_apply_smote()

        sc=StandardScaler()
        self.X_train=sc.fit_transform(self.X_train)
        self.X_test=sc.transform(self.X_test)
        self.estimator = KNeighborsClassifier(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
