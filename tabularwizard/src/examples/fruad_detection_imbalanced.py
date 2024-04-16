# https://www.kaggle.com/code/tboyle10/methods-for-dealing-with-imbalanced-data/notebook
from datetime import datetime
import os
import pickle
import pandas as pd
from tabularwizard.src.classification.evaluate import Evaluate
from tabularwizard.src.classification.model.catboot_classifier import CatboostClassifier
from tabularwizard.src.classification.model.mlpclassifier import MLPNetClassifier
from tabularwizard.src.classification.model.lightgbm_classifier import LightgbmClassifier
from tabularwizard.src.classification.model.random_forest_classifier import RandomForestClassifierCustom
from tabularwizard.src.classification.model.xgboost_classifier import XgboostClassifier
from tabularwizard.src.data_preprocessing import DataPreprocessing
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', 'classification', f"fraud_lightgbm_{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)
SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'finalized_lgbm_model.sav')
SAVED_MODEL_EVALUATION = os.path.join(SAVED_MODEL_FOLDER, 'model_eval')

def train_model():
   start_time = datetime.now().strftime("%H:%M:%S")
   df = pd.read_csv('tabularwizard/datasets/credit_fraud/creditcard.csv')

   print(df.shape)
   print(df.Class.value_counts())
   print((len(df.loc[df.Class==1])) / (len(df.loc[df.Class == 0])))

   dataPreprocessing  = DataPreprocessing()

    # underampling_df = dataPreprocessing.majority_minority_class(df, "Class", 1, 0)
    # print(underampling_df.Class.value_counts())
   
   classifier = LightgbmClassifier(train_df = df, prediction_column = 'Class')
   # df = dataPreprocessing.majority_minority_class(pd.concat([classifier.X_train, classifier.y_train], axis=1), "Class", 1, 0)
   
    # classifier.y_train = sampling_df.Class
    # classifier.X_train = sampling_df.drop('Class', axis=1)
    # print(sampling_df.Class.value_counts())
   #  dataPreprocessing.oversampling_minority_classifier(classifier, 1, 0)
   classifier.tune_hyper_parameters()
   model = classifier.train()
   pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))

   evaluate = Evaluate()
   evaluations = evaluate.evaluate_train_and_test(model, classifier)
    
   print(f"model evaluations: {evaluate.format_train_and_test_evaluation(evaluations)}")
   end_time = datetime.now().strftime("%H:%M:%S")
   print("start time =", start_time)
   print("end time =", end_time)

   #  with open(SAVED_MODEL_EVALUATION, 'w') as file:
   #      file.write(evaluations)

def use_traned_model():
    pass

if __name__ == '__main__':
    train_model()
    use_traned_model()


'''
oversampling:
model evaluations: Train eval:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    199008
           1       1.00      1.00      1.00    199008

    accuracy                           1.00    398016
   macro avg       1.00      1.00      1.00    398016
weighted avg       1.00      1.00      1.00    398016

confusion_matrix is [[198998     10]
 [     0 199008]]
score: 1.0
Test eval:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.92      0.88      0.89       136

    accuracy                           1.00     85443
   macro avg       0.96      0.94      0.95     85443
weighted avg       1.00      1.00      1.00     85443

confusion_matrix is [[85296    11]
 [   17   119]]
score: 1.0
'''


'''
undersampleling:
model evaluations: Train eval:
              precision    recall  f1-score   support

           0       0.50      1.00      0.67       356
           1       0.00      0.00      0.00       356

    accuracy                           0.50       712
   macro avg       0.25      0.50      0.33       712
weighted avg       0.25      0.50      0.33       712

confusion_matrix is [[356   0]
 [356   0]]
score: 0.5
Test eval:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.00      0.00      0.00       136

    accuracy                           1.00     85443
   macro avg       0.50      0.50      0.50     85443
weighted avg       1.00      1.00      1.00     85443

confusion_matrix is [[85307     0]
 [  136     0]]
score: 0.998
'''