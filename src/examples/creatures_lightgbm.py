# https://www.kaggle.com/code/mrtgocer/from-zero-to-hero-lightgbm-classifier
from datetime import datetime
import os
import pickle
import pandas as pd
from src.classification.evaluate import Evaluate
from src.classification.model.lightgbm_classifier import LightgbmClassifier
from src.data_preprocessing import DataPreprocessing



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', 'classification', f"creatures_lightgbm_{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)
SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'finalized_lgbm_model.sav')
SAVED_MODEL_EVALUATION = os.path.join(SAVED_MODEL_FOLDER, 'model_eval')
train_path = "datasets/ghouls-goblins-and-ghosts-boo/train.csv"
test_path = "datasets/ghouls-goblins-and-ghosts-boo/test.csv"

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

    lgbm_classifier = LightgbmClassifier(train_df = train_data, prediction_column = 'type')
    lgbm_classifier.tune_hyper_parameters(scoring='accuracy')
    model = lgbm_classifier.train()
    pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))

    evaluate = Evaluate()
    evaluations = evaluate.evaluate_train_and_test(model, lgbm_classifier)
    
    print(f"model evaluations: {evaluations}")
    with open(SAVED_MODEL_EVALUATION, 'w') as file:
        file.write(evaluations)

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