from sklearn.neighbors import KNeighborsClassifier
from tabularwizard.src.classification.model.base_classifier_model import BaseClassfierModel
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Categorical, Integer


DEFAULT_PARAMS = {
    'n_neighbors': Integer(1, 30),  # Number of neighbors
    'weights': Categorical(['uniform', 'distance']),  # Weight type
    'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),  # Algorithm used to compute the nearest neighbors
    'leaf_size': Integer(20, 50),  # Leaf size passed to BallTree or KDTree
     #'p': Categorical([1, 2])  # Power parameter for the Minkowski metric
}

class KnnClassifier(BaseClassfierModel):
    def __init__(self, train_df, prediction_column, split_column=None, test_size=0.3, already_splited_data=None, *args, **kwargs):
        super().__init__(train_df, prediction_column, split_column=split_column, test_size=test_size, already_splited_data=already_splited_data)
        self.unique_classes = train_df[prediction_column].nunique()
        self.check_and_apply_smote()

        sc=StandardScaler()
        self.X_train=sc.fit_transform(self.X_train)
        self.X_test=sc.transform(self.X_test)
        self.estimator = KNeighborsClassifier(*args, **kwargs)

    @property
    def default_params(self):
        return DEFAULT_PARAMS
