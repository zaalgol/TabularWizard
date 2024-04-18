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

        return {'mse':mse, 'rmse':rmse,'r2':r2, 'mae':mae, 'rmsle':rmsle }

        # Construct results string
        # results_lines = {
        #     'mae': f"Mean Absolute Error - mae: {mae}",
        #     'mse': f"Mean Squared Error - mse: {mse}",
        #     'rmse': f"Root Mean Squared Error - rmse: {rmse}",
        #     'r2': f"R2 Score: {r2}",
        #     'rmsle': f"Root Mean Squared Logarithmic Error - rmsle: {rmsle}"
        # }

        # if title is not None:
        #     results_lines[title]= title

        # return results_lines
    
    def evaluate_train_and_test(self, model, regressor):
        y_train_predict = self.predict(model, regressor.X_train)
        train_evaluations = self.evaluate_predictions(regressor.y_train, y_train_predict, "Train evaluation:")
        # model["train_evaluation"] = train_evaluation

        y_test_predict = self.predict(model, regressor.X_test)
        test_evaluations = self.evaluate_predictions(regressor.y_test, y_test_predict, "Test evaluation:")

        return {'train_evaluations':train_evaluations, 'test_evaluations': test_evaluations}
        # model["test_evaluation"] = test_evaluation

    
    def format_train_and_test_evaluation(self, evaluations):
         return "\n".join([
            "\nTrain eval:\n {}", 
            "{}", 
            "\nTest eval:\n {}", 
            "\n"
        ]).format(str(evaluations['train_evaluations'].values()), "*" * 100, str(evaluations['test_evaluations'].values()))



            