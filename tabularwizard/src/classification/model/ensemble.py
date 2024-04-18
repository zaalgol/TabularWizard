# https://chat.openai.com/c/60d698a6-9405-4db9-8831-6b2491bb9111
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.ensemble import VotingClassifier
from itertools import islice


from tabularwizard.src.classification.model.base_classifier_model import BaseClassfierModel
from src.classification.model.knn_classifier import KnnClassifier
from src.classification.model.logistic_regression import LRegression
from src.classification.model.mlpclassifier import MLPNetClassifier
from src.classification.model.lightgbm_classifier import LightgbmClassifier
from src.classification.model.random_forest_classifier import RandomForestClassifierCustom
from src.classification.model.xgboost_classifier import XgboostClassifier
from src.classification.evaluate import Evaluate
from src.data_preprocessing import DataPreprocessing

class Ensemble(BaseClassfierModel):
    def __init__(self, train_df, prediction_column, split_column=None, test_size=0.3, scoring='accuracy'):
        self.classifiers = {}
        super().__init__(train_df=train_df, prediction_column=prediction_column, scoring=scoring, split_column=split_column, test_size=test_size)
        self.already_splited_data = {'X_train': self.X_train, 'X_test': self.X_test, 'y_train': self.y_train, 'y_test':self.y_test}
        self.evaluate = Evaluate()

    def create_models(self, df):
        self.classifiers['lgbm_classifier'] = {'model':LightgbmClassifier(train_df = df.copy(), prediction_column = self.prediction_column, already_splited_data=self.already_splited_data)}
        self.classifiers['xgb_classifier'] = {'model':XgboostClassifier(train_df = df.copy(), prediction_column = self.prediction_column, already_splited_data=self.already_splited_data)}
        self.classifiers['knn_classifier'] = {'model':KnnClassifier(train_df = df.copy(), prediction_column = self.prediction_column, already_splited_data=self.already_splited_data)}
        self.classifiers['LRegression'] = {'model':LRegression(train_df = df.copy(), prediction_column = self.prediction_column, already_splited_data=self.already_splited_data)}
        self.classifiers['mlp_classifier'] = {'model':MLPNetClassifier(train_df = df.copy(), prediction_column = self.prediction_column, already_splited_data=self.already_splited_data)}
        
    def tune_hyper_parameters(self):
        for classifier_value in self.classifiers.values():
            classifier_value['model'].tune_hyper_parameters(scoring=self.scoring)

    def train_all_models(self):
        for classifier_value in self.classifiers.values():
            classifier_value['trained_model'] = classifier_value['model'].train()

    def evaluate_all_models(self):
        for classifier_value in self.classifiers.values():
            classifier_value['evaluations'] = self.evaluate.evaluate_train_and_test(classifier_value['trained_model'], classifier_value['model'])
        self.classifiers= dict(sorted(self.classifiers.items(), key=lambda item: item[1]['evaluations']['test_metrics'][self.scoring], reverse=True))

    def create_voting_classifier(self, top_n_best_models=3):
        model_list = [(name, info['model'].estimator) for name, info in islice(self.classifiers.items(), top_n_best_models)]
        self.voting_classifier = VotingClassifier(estimators=model_list, voting='soft')

    def train_voting_classifier(self):
        self.trained_voting_classifier = self.voting_classifier.fit(self.X_train, self.y_train)

    def evaluate_voting_classifier(self):
        self.voting_classifier_evaluations = self.evaluate.evaluate_train_and_test(self.trained_voting_classifier, self)

    def hard_predict(self, trained_models):
        predictions = [model.predict(X_test) for model, X_test in trained_models]

        # Use mode to find the most common class label
        predictions = np.array(predictions)
        return mode(predictions, axis=0)[0].flatten()
    
    def soft_predict(self, trained_models):
        if all(hasattr(model, 'predict_proba') for model, _ in trained_models):
            # Predict probabilities
            probabilities = [model.predict_proba(X_test) for model, X_test  in trained_models]

            # Average the probabilities
            avg_probabilities = np.mean(probabilities, axis=0)

            # Predict the class with the highest average probability
            soft_vote_predictions = np.argmax(avg_probabilities, axis=1)
        else:
            print("Not all models support predict_proba method necessary for soft voting.")



# # train_path = "tabularwizard/datasets/phone-price-classification/train.csv"
# train_path = "tabularwizard/datasets/titanic.csv"
# train_data = pd.read_csv(train_path)


# num_rows = train_data.shape[0]
# num_common = int(num_rows * 0.48)
# num_rare = int(num_rows * 0.02)

# # Generate a list with the desired distribution of values
# values = ['common'] * num_common + ['rare'] * num_rare
# values += [np.nan] * (num_rows - len(values))  # Fill the rest with NaN

# # Shuffle the list to distribute 'common' and 'rare' randomly across the column
# np.random.shuffle(values)
# train_data['temp'] = values

# train_data['Name2'] = train_data['Name'] 
# data_preprocessing = DataPreprocessing()
# train_data = data_preprocessing.sanitize_dataframe(train_data)
# train_data = data_preprocessing.fill_missing_numeric_cells(train_data)
# encoding_rules = data_preprocessing.create_encoding_rules(train_data)
# train_data = data_preprocessing.apply_encoding_rules(train_data, encoding_rules)
# ensemble = Ensemble(train_df=train_data, prediction_column='Survived')
# ensemble.create_models(train_data)
# ensemble.train_all_models()
# ensemble.evaluate_all_models()

# ensemble.create_voting_classifier()
# ensemble.train_voting_classifier()
# ensemble.evaluate_voting_classifier()

t=0




        


    