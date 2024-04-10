
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class Evaluate:
    def predict(self, model, X_data):
        return model.predict(X_data)
    
    def evaluate_classification (self, y_true, y_predict):
        cm = confusion_matrix (y_true, y_predict)
        print_str =  "\n".join([classification_report(y_true, y_predict), f"confusion_matrix: \n {cm}"])
        # print (print_str)
        return print_str
        
    
    def evaluate_train_and_test(self, model, classifier):
        y_predict = self.predict(model, classifier.X_train)
        train_score = round(accuracy_score(classifier.y_train, y_predict), 3)
        train_evaluations = self.evaluate_classification (classifier.y_train, y_predict)

        y_predict = self.predict(model, classifier.X_test)
        test_score = round(accuracy_score(classifier.y_test, y_predict), 3)
        test_evaluations = self.evaluate_classification (classifier.y_test, y_predict)

        # return "\n".join(["\nTrain eval:",  str(train_evaluations), f'score: {train_score}',  "\nTest eval:", str(test_evaluations), f'score: {test_score}', "*" * 100, "\n"])
        return train_evaluations, train_score, test_evaluations, test_score
    
    def print_train_and_test_evaluation(self, train_evaluations, train_score, test_evaluations, test_score):
         print("\n".join([
            "\nTrain eval:\n {}", 
            "Train score: {}", 
            "{}", 
            "\nTest eval:\n {}", 
            "Test score: {}", 
            "\n"
        ]).format(str(train_evaluations), train_score, "*" * 100, str(test_evaluations), test_score))
    