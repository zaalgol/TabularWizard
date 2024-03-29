# https://www.kaggle.com/code/mrtgocer/from-zero-to-hero-lightgbm-classifier
from datetime import datetime
import os
import pickle
import pandas as pd
from tabularwizard.src.classification.evaluate import Evaluate
from tabularwizard.src.classification.model.mlpclassifier import MLPNetClassifier
from tabularwizard.src.classification.model.lightgbm_classifier import LightgbmClassifier
from tabularwizard.src.classification.model.random_forest_classifier import RandomForestClassifierCustom
from tabularwizard.src.classification.model.xgboost_classifier import XgboostClassifier
from tabularwizard.src.data_preprocessing import DataPreprocessing
from sklearn.ensemble import VotingClassifier



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', 'classification', f"creatures_lightgbm_{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)
LGBM_SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'finalized_lgbm_model.sav')
XGB_SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'finalized_xbm_model.sav')
RF_SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'finalized_rf_model.sav')
LGBM_SAVED_MODEL_FILE2 = os.path.join(SAVED_MODEL_FOLDER, 'finalized_lgbm_model2.sav')
XGB_SAVED_MODEL_FILE2 = os.path.join(SAVED_MODEL_FOLDER, 'finalized_xbm_model2.sav')
RF_SAVED_MODEL_FILE2 = os.path.join(SAVED_MODEL_FOLDER, 'finalized_rf_model2.sav')
SAVED_MODEL_EVALUATION = os.path.join(SAVED_MODEL_FOLDER, 'model_eval')
train_path = "tabularwizard/datasets/ghouls-goblins-and-ghosts-boo/train.csv"
test_path = "tabularwizard/datasets/ghouls-goblins-and-ghosts-boo/test.csv"

dataPreprocessing  = DataPreprocessing()

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

def use_traned_model():
    pass


def train_model():
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print(train_data.head())
    print(test_data.head())

    dataPreprocessing.describe_datafranme(train_data)
    print(dataPreprocessing.get_missing_values_per_coloun(train_data))
    print(dataPreprocessing.get_missing_values_per_coloun(test_data))

    train_data = dataPreprocessing.one_hot_encode_column(train_data, 'color')
    train_data = dataPreprocessing.map_order_column(train_data, 'type', {"Ghoul":1, "Goblin":2, "Ghost":0})
    train_data = train_data.set_index('id')
    print(train_data.head())

    # results = []

    mlp_classifier = MLPNetClassifier(train_df = train_data.copy(), prediction_column = 'type')
    mlp_classifier.tune_hyper_parameters(scoring='accuracy')
    mlp_model = mlp_classifier.train()

    # pickle.dump(rf_model, open(RF_SAVED_MODEL_FILE, 'wb'))

    rf_classifier = RandomForestClassifierCustom(train_df = train_data.copy(), prediction_column = 'type')
    rf_classifier.tune_hyper_parameters(scoring='accuracy')
    rf_model = rf_classifier.train()
    pickle.dump(rf_model, open(RF_SAVED_MODEL_FILE, 'wb'))

    lgbm_classifier = LightgbmClassifier(train_df = train_data.copy(), prediction_column = 'type')
    lgbm_classifier.tune_hyper_parameters(scoring='accuracy')
    lgbm_model = lgbm_classifier.train()
    pickle.dump(lgbm_model, open(LGBM_SAVED_MODEL_FILE, 'wb'))

    xgb_classifier = XgboostClassifier(train_df = train_data.copy(), prediction_column = 'type')
    xgb_classifier.tune_hyper_parameters(scoring='accuracy')
    xgb_model = xgb_classifier.train()
    pickle.dump(xgb_model, open(XGB_SAVED_MODEL_FILE, 'wb'))


    rf_classifier2 = RandomForestClassifierCustom(train_df = train_data.copy(), prediction_column = 'type')
    rf_classifier2.tune_hyper_parameters(scoring='accuracy')
    rf_model2 = rf_classifier2.train()
    pickle.dump(rf_model2, open(RF_SAVED_MODEL_FILE2, 'wb'))

    lgbm_classifier2 = LightgbmClassifier(train_df = train_data.copy(), prediction_column = 'type')
    lgbm_classifier2.tune_hyper_parameters(scoring='accuracy')
    lgbm_model2 = lgbm_classifier2.train()
    pickle.dump(lgbm_model2, open(LGBM_SAVED_MODEL_FILE2, 'wb'))

    xgb_classifier2 = XgboostClassifier(train_df = train_data.copy(), prediction_column = 'type')
    xgb_classifier2.tune_hyper_parameters(scoring='accuracy')
    xgb_model2 = xgb_classifier2.train()
    pickle.dump(xgb_model2, open(XGB_SAVED_MODEL_FILE2, 'wb'))

  
    
    estimators = [('lgbm_model2', lgbm_model2), ('xgb_model2', xgb_model2), ('rf_model2', rf_model2), \
                   ('lgbm_model', lgbm_model), ('xgb_model', xgb_model), ('rf_model', rf_model), \
                   ('mlp_model', mlp_model)]
    ensemble = VotingClassifier(estimators)
    ensemble_model = ensemble.fit(rf_classifier2.X_train, rf_classifier2.y_train)
    evaluate = Evaluate()

    lgbm_evaluations = evaluate.evaluate_train_and_test(lgbm_model, lgbm_classifier)
    xgb_evaluations = evaluate.evaluate_train_and_test(xgb_model, xgb_classifier)
    rf_evaluations = evaluate.evaluate_train_and_test(rf_model, rf_classifier)
    lgbm_evaluations2 = evaluate.evaluate_train_and_test(lgbm_model2, lgbm_classifier2)
    xgb_evaluations2 = evaluate.evaluate_train_and_test(xgb_model2, xgb_classifier2)
    rf_evaluations2 = evaluate.evaluate_train_and_test(rf_model2, rf_classifier2)
    mlp_evaluations = evaluate.evaluate_train_and_test(mlp_model, mlp_classifier)
    ensemble_evaluations = evaluate.evaluate_train_and_test(ensemble_model, rf_classifier2)
    
    print("*" * 500 + f"lgbm_evaluations: {lgbm_evaluations} + best_score_: {lgbm_model.best_score_}")
    print("*" * 500 + f"xgb_evaluations: {xgb_evaluations} + best_score_: {xgb_model.best_score_}")
    print("*" * 500 + f"rf_evaluations: {rf_evaluations} + best_score_: {rf_model.best_score_}")
    print("*" * 500 + f"lgbm_evaluations2: {lgbm_evaluations2} + best_score_: {lgbm_model2.best_score_}")
    print("*" * 500 + f"xgb_evaluations2: {xgb_evaluations2} + best_score_: {xgb_model2.best_score_}")
    print("*" * 500 + f"rf_evaluations2: {rf_evaluations2} + best_score_: {rf_model2.best_score_}")
    
    print("*" * 500 + f"mlp_evaluations: {mlp_evaluations} + best_score_: {mlp_model.best_score_}")
    print("*" * 500 + f"ensemble_evaluations: {ensemble_evaluations}")
    t=0



    # with open(SAVED_MODEL_EVALUATION, 'w') as file:
    #     file.write(lgbm_evaluations)

    # for i in range(10):
    #     lgbm_classifier = LightgbmClassifier(train_df = train_data, prediction_column = 'type')
    #     lgbm_classifier.tune_hyper_parameters(scoring='accuracy')
    #     result, best_params, cv_score, test_score = lgbm_classifier.train()
    #     # Storing the results of this iteration
    #     iteration_results = {
    #         "best_params": best_params,
    #         "cv_score": cv_score,
    #         "test_score": test_score
    #     }
    #     results.append(iteration_results)


    # Printing the results after the loop
    # print(f"*" * 100)
    # for idx, res in enumerate(results, 1):
    #     print(f"Iteration {idx}:")
    #     print(f"Best Params: {res['best_params']}")
    #     print(f"CV Score: {res['cv_score']}")
    #     print(f"Test Score: {res['test_score']}")
    #     print("-" * 50)




if __name__ == '__main__':
    train_model()
    use_traned_model()