import numpy as np
from scipy.stats import mode

class Ensemble:
    def hard_predict(self, models_data):
        predictions = [model.predict(X_test) for model, X_test in models_data]

        # Use mode to find the most common class label
        predictions = np.array(predictions)
        return mode(predictions, axis=0)[0].flatten()
    
    def soft_predict(self, models_data):
        if all(hasattr(model, 'predict_proba') for model, _ in models_data):
            # Predict probabilities
            probabilities = [model.predict_proba(X_test) for model, X_test  in models_data]

            # Average the probabilities
            avg_probabilities = np.mean(probabilities, axis=0)

            # Predict the class with the highest average probability
            soft_vote_predictions = np.argmax(avg_probabilities, axis=1)
        else:
            print("Not all models support predict_proba method necessary for soft voting.")