import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error

class Evaluate:
    def predict(self, model, X_data):
        return model.predict (X_data)
    
    def evaluate_predictions (self, y_true, y_pred, title=None):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)  # Calculating RMSE
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))


        # Construct results string
        results_lines = [
            f"Mean Absolute Error - mae: {mae}",
            f"Mean Squared Error - mse: {mse}",
            f"Root Mean Squared Error - rmse: {rmse}",
            f"R2 Score: {r2}",
            f"Root Mean Squared Logarithmic Error - rmsle: {rmsle}"
        ]

        if title is not None:
            results_lines.insert(0, title)

        results = "\n".join(results_lines)

        print(results)

        return results + "\n\n"
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        y_train_predict = self.predict(model, X_train)
        train_evaluation = self.evaluate_predictions(y_train, y_train_predict, "Train evaluation:")
        # model["train_evaluation"] = train_evaluation

        y_test_predict = self.predict(model, X_test)
        test_evaluation = self.evaluate_predictions(y_test, y_test_predict, "Test evaluation:")

        return train_evaluation + test_evaluation
        # model["test_evaluation"] = test_evaluation



            