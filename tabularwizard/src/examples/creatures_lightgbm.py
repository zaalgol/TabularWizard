# https://www.kaggle.com/code/mrtgocer/from-zero-to-hero-lightgbm-classifier
from datetime import datetime
import os
import pickle
import pandas as pd
from tabularwizard.src.classification.evaluate import Evaluate
from tabularwizard.src.classification.model.lightgbm_classifier import LightgbmClassifier
from tabularwizard.src.classification.model.logistic_regression import LRegression
from tabularwizard.src.classification.model.knn_classifier import KnnClassifier
from tabularwizard.src.data_preprocessing import DataPreprocessing
from sklearn.preprocessing import StandardScaler



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', 'classification', f"creatures_lightgbm_{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)
SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'finalized_lgbm_model.sav')
SAVED_MODEL_EVALUATION = os.path.join(SAVED_MODEL_FOLDER, 'model_eval')
p = os.getcwd()
train_path = "tabularwizard/datasets/ghouls-goblins-and-ghosts-boo/train.csv"
test_path = "tabularwizard/datasets/ghouls-goblins-and-ghosts-boo/test.csv"

data_preprocessing  = DataPreprocessing()

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

def use_traned_model():
    pass

params = {'colsample_bytree': 0.09854898073399448, 'learning_rate': 0.020462709433918657,
            'max_bin': 1000, 'max_depth': 256, 'min_child_samples': 24, 'min_child_weight': 0.01,
            'n_estimators': 992, 'num_leaves': 2,
            'reg_alpha': 0.5, 'reg_lambda': 0.5, 'subsample': 0.9293522541996262, 'subsample_freq': 10}

def train_model():
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print(train_data.head())
    print(test_data.head())

    data_preprocessing.describe_datafranme(train_data)
    print(data_preprocessing.get_missing_values_per_coloun(train_data))
    print(data_preprocessing.get_missing_values_per_coloun(test_data))

    # train_data = dataPreprocessing.one_hot_encode_column(train_data, 'color')
    # train_data = dataPreprocessing.map_order_column(train_data, 'type', {"Ghoul":1, "Goblin":2, "Ghost":0})
    train_data = train_data.set_index('id')
    print(train_data.head())

    results = {}
    start_time = datetime.now().strftime("%H:%M:%S")

    df = data_preprocessing.one_hot_encode_column(train_data, "color")
    lRegression = LRegression(train_df = df, target_column = 'type')
    lRegression.tune_hyper_parameters()
    trained_lRegression = lRegression.train()

    evaluate = Evaluate()
    evaluations = evaluate.evaluate_train_and_test(trained_lRegression, lRegression)
    evaluate.format_train_and_test_evaluation(evaluations)
    results['lRegression'] = evaluations

    knnClassifier = KnnClassifier(train_df = df, target_column = 'type')
    knnClassifier.tune_hyper_parameters()
    trained_knnClassifier = knnClassifier.train()
    evaluate = Evaluate()
    evaluations = evaluate.evaluate_train_and_test(trained_knnClassifier, knnClassifier)
    evaluate.format_train_and_test_evaluation(evaluations)
    results['knnClassifier'] = evaluations

    # lgbm_classifier = LightgbmClassifier(train_df = train_data, target_column = 'type', device='gpu')
    cat_features  =  data_preprocessing.get_all_categorical_columns_names(train_data)
    for feature in cat_features:
        train_data[feature] = train_data[feature].astype('category')
    lgbm_classifier = LightgbmClassifier(train_df = train_data, target_column = 'type')
    lgbm_classifier.tune_hyper_parameters()
    trained_lgbm = lgbm_classifier.train()


    # pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))

    evaluate = Evaluate()
    evaluations = evaluate.evaluate_train_and_test(trained_lgbm, lgbm_classifier)
    evaluate.format_train_and_test_evaluation(evaluations)
    results['lightbm'] = evaluations
    end_time = datetime.now().strftime("%H:%M:%S")
    print("start time =", start_time)
    print("end time =", end_time)
    # print(f"model evaluations: {evaluations}")
    # with open(SAVED_MODEL_EVALUATION, 'w') as file:
    #     file.write(str(evaluations))

    # lgbm_classifier.save_feature_importances(model_folder=SAVED_MODEL_FOLDER)
    # lgbm_classifier.save_tree_diagram(tree_index=0, model_folder=SAVED_MODEL_FOLDER)

    # for i in range(10):
    #     lgbm_classifier = LightgbmClassifier(train_df = train_data, target_column = 'type')
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