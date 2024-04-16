
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss, precision_score, recall_score, auc

class Evaluate:
    def predict(self, model, X_data):
        return model.predict(X_data)
    
    def get_confution_matrix (self, y_true, y_predict):
        return confusion_matrix (y_true, y_predict)
        # print_str =  "\n".join([classification_report(y_true, y_predict), f"confusion_matrix: \n {cm}"])
        # # print (print_str)
        # return print_str

    def get_accurecy_score(self, y_true, y_predict):
        return round(accuracy_score(y_true, y_predict), 4)


    def evaluate_train_and_test(self, model, classifier):
        y_predict = self.predict(model, classifier.X_train)
        train_score = self.get_accurecy_score(classifier.y_train, y_predict)
        train_evaluations = self.get_confution_matrix (classifier.y_train, y_predict)
        train_evaluations_str = "\n".join([classification_report(classifier.y_train, y_predict), f"confusion_matrix: \n {train_evaluations}"])

        y_predict = self.predict(model, classifier.X_test)
        test_score =  self.get_accurecy_score(classifier.y_test, y_predict)
        test_evaluations = self.get_confution_matrix (classifier.y_test, y_predict)
        test_evaluations_str = "\n".join([classification_report(classifier.y_test, y_predict), f"confusion_matrix: \n {test_evaluations}"])

        # return "\n".join(["\nTrain eval:",  str(train_evaluations), f'score: {train_score}',  "\nTest eval:", str(test_evaluations), f'score: {test_score}', "*" * 100, "\n"])
        return {'train_evaluations': train_evaluations_str, 'train_score':train_score, 'test_evaluations':test_evaluations_str, 'test_score': test_score}
    
    def format_train_and_test_evaluation(self, evaluations):
         return "\n".join([
            "\nTrain eval:\n {}", 
            "Train score: {}", 
            "{}", 
            "\nTest eval:\n {}", 
            "Test score: {}", 
            "\n"
        ]).format(str(evaluations['train_evaluations']), evaluations['train_score'], "*" * 100, str(evaluations['test_evaluations']), evaluations['test_score'])
    