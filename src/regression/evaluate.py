import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error

class Evaluate:
    def predict(self, model, X_data):
        return model.predict (X_data)
    
    def evaluate_predictions (self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))


        # print scores
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        print("R2 Score:", r2)
        print("rmsle:", rmsle)
            