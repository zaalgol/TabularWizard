import os
from sklearn.neural_network import MLPClassifier
from src.classification.model.base_classifier_model import BaseClassfierModel
import matplotlib.pyplot as plt

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
    
    def visualize_weights(self,  model_folder='', filename='mlp_weights_visualization.png'):
        layers = len(self.search.best_estimator_.coefs_)
        fig, axes = plt.subplots(nrows=1, ncols=layers, figsize=(20, 5))
        for i in range(layers):
            ax = axes[i]
            ax.matshow(self.search.best_estimator_.coefs_[i], cmap='viridis')
            ax.set_title(f'Layer {i+1}')
            ax.set_xlabel('Neurons in Layer')
            ax.set_ylabel('Input Features')
        plt.tight_layout()
        plt.savefig(os.path.join(model_folder, filename))
        plt.close(fig)
