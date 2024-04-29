# https://www.kaggle.com/code/mrtgocer/from-zero-to-hero-lightgbm-classifier
from datetime import datetime
import os
import pickle
import pandas as pd
from src.classification.model.knn_classifier import KnnClassifier
from src.classification.model.logistic_regression import LRegression
from src.classification.evaluate import Evaluate
from src.classification.model.mlpclassifier import MLPNetClassifier
from src.classification.model.lightgbm_classifier import LightgbmClassifier
from src.classification.model.random_forest_classifier import RandomForestClassifierCustom
from src.classification.model.xgboost_classifier import XgboostClassifier
from src.data_preprocessing import DataPreprocessing
from src.classification.model.ensemble import Ensemble
from sklearn.ensemble import VotingClassifier



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

iris_path = 'tabularwizard/datasets/titanic.csv'

data_preprocessing  = DataPreprocessing()

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

def use_traned_model():
    pass


def train_model():
    start_time = datetime.now().strftime("%H:%M:%S")
    tune = False
    target_column = 'Survived'
    df = pd.read_csv(iris_path)

    print(df.head())


    data_preprocessing.describe_datafranme(df)
    df = data_preprocessing.sanitize_column_names(df)
    df = data_preprocessing.exclude_non_numeric_columns_with_large_unique_values(df, 10)
    print(data_preprocessing.get_missing_values_per_coloun(df))
    df = data_preprocessing.one_hot_encode_all_categorical_columns(df)
    df = data_preprocessing.fill_missing_numeric_cells(df)
    # train_data = dataPreprocessing.one_hot_encode_column(train_data, 'color')
    # train_data = train_data.set_index('id')
    print(df.head())

    mlp_classifier = MLPNetClassifier(train_df = df.copy(), target_column = target_column)
    if tune:
        mlp_classifier.tune_hyper_parameters(scoring='accuracy')
    mlp_model = mlp_classifier.train()


    rf_classifier = RandomForestClassifierCustom(train_df = df.copy(), target_column = target_column)
    if tune:
        rf_classifier.tune_hyper_parameters(scoring='accuracy')
    rf_model = rf_classifier.train()


    lgbm_classifier = LightgbmClassifier(train_df = df.copy(), target_column = target_column)
    if tune:
        lgbm_classifier.tune_hyper_parameters(scoring='accuracy')
    lgbm_model = lgbm_classifier.train()


    xgb_classifier = XgboostClassifier(train_df = df.copy(), target_column = target_column)
    if tune:
        xgb_classifier.tune_hyper_parameters(scoring='accuracy')
    xgb_model = xgb_classifier.train()
 

    lr_classifier = LRegression(train_df = df.copy(), target_column = target_column)
    if tune:
        lr_classifier.tune_hyper_parameters(scoring='accuracy')
    lr_model = lr_classifier.train()

    knn_classifier = KnnClassifier(train_df = df.copy(), target_column = target_column)
    if tune:
        lr_classifier.tune_hyper_parameters(scoring='accuracy')
    knn_model = knn_classifier.train()

    # estimators = [ ('lgbm_model', lgbm_model), ('xgb_model', xgb_model), ('rf_model', rf_model), \
    #                ('mlp_model', mlp_model)]
    # ensemble = VotingClassifier(estimators)
    # ensemble_model = ensemble.fit(rf_classifier.X_train, rf_classifier.y_train)
    ensemble = Ensemble()
    model_data = [(mlp_model, mlp_classifier.X_test), (rf_model, rf_classifier.X_test),
                  (lgbm_model, lgbm_classifier.X_test), (xgb_model, xgb_classifier.X_test),
                  (lr_model, lr_classifier.X_test), (knn_model, knn_classifier.X_test)]
    ensemble_hard_predict_model = ensemble.hard_predict(model_data)
    # ensemble_soft_predict_model = ensemble.soft_predict(model_data)
    evaluate = Evaluate()

    lgbm_evaluations = evaluate.evaluate_train_and_test(lgbm_model, lgbm_classifier)
    xgb_evaluations = evaluate.evaluate_train_and_test(xgb_model, xgb_classifier)
    rf_evaluations = evaluate.evaluate_train_and_test(rf_model, rf_classifier)
    mlp_evaluations = evaluate.evaluate_train_and_test(mlp_model, mlp_classifier)
    lr_evaluations = evaluate.evaluate_train_and_test(lr_model, lr_classifier)
    knn_evaluations = evaluate.evaluate_train_and_test(knn_model, knn_classifier)
    ensemble_hard_evaluation = evaluate.get_accurecy_score(ensemble_hard_predict_model, xgb_classifier.y_test) # any classifier. need it only for the X and y
    # ensemble_soft_evaluation = evaluate.get_accurecy_score(ensemble_soft_predict_model, xgb_classifier.y_test)
    
    print("*" * 200 + f" lgbm_evaluations: {evaluate.format_train_and_test_evaluation(lgbm_evaluations)}")
    print("*" * 200 + f" xgb_evaluations: {evaluate.format_train_and_test_evaluation(xgb_evaluations)}")
    print("*" * 200 + f" rf_evaluations: {evaluate.format_train_and_test_evaluation(rf_evaluations)}")
    print("*" * 200 + f" mlp_evaluations: {evaluate.format_train_and_test_evaluation(mlp_evaluations)}")
    print("*" * 200 + f" lr_evaluations: {evaluate.format_train_and_test_evaluation(lr_evaluations)}")
    print("*" * 200 + f" knn_evaluations: {evaluate.format_train_and_test_evaluation(knn_evaluations)}")

    print("#" * 200 + f" lgbm_evaluations score: {lgbm_evaluations['test_score']}")
    print("#" * 200 + f" xgb_evaluations score: {xgb_evaluations['test_score']}")
    print("#" * 200 + f" rf_evaluations score: {rf_evaluations['test_score']}")
    print("#" * 200 + f" mlp_evaluations score: {mlp_evaluations['test_score']}")
    print("#" * 200 + f" lr_evaluations score: {lr_evaluations['test_score']}")
    print("#" * 200 + f" knn_evaluations score: {knn_evaluations['test_score']}")
    print("#" * 200 + f" ensemble_hard_evaluations score: {ensemble_hard_evaluation}")
    print("#" * 200 + f" : tune= {tune}")
    end_time = datetime.now().strftime("%H:%M:%S")
    print("start time =", start_time)
    print("end time =", end_time)
    # print(f"model evaluations: {evalua
    # print("#" * 200 + f" ensemble_soft_evaluations score: {ensemble_soft_evaluation}")



    
    # print("*" * 200 + f"ensemble_evaluations: {ensemble_evaluations}")
    t=0



    # with open(SAVED_MODEL_EVALUATION, 'w') as file:
    #     file.write(lgbm_evaluations)

    # for i in range(10):
    #     lgbm_classifier = LightgbmClassifier(train_df = train_data, target_column = target_column)
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