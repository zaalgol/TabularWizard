
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class Evaluate:
    def predict(self, model, X_data):
        return model.predict (X_data)
    
    def evaluate_predictions (self, y_true, y_predict):
        cm = confusion_matrix (y_true, y_predict)
        print (classification_report (y_true, y_predict))
        print (f"confusion_matrix is {cm}")
    
        