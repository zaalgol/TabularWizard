import os
import pickle
import pandas as pd
from src.classification.evaluate import Evaluate
from src.classification.model.mlpclassifier import MLPNetClassifier
from src.data_preprocessing import DataPreprocessing
from datetime import datetime



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVED_MODEL_FOLDER = os.path.join('results', 'trained_models', 'classification', f"creatures_mpl_{timestamp}")
os.makedirs(SAVED_MODEL_FOLDER)
SAVED_MODEL_FILE = os.path.join(SAVED_MODEL_FOLDER, 'finalized_lgbm_model.sav')
SAVED_MODEL_EVALUATION = os.path.join(SAVED_MODEL_FOLDER, 'model_eval')
train_path = "tabularwizard/datasets/ghouls-goblins-and-ghosts-boo/train.csv"
test_path = "tabularwizard/datasets/ghouls-goblins-and-ghosts-boo/test.csv"

dataPreprocessing  = DataPreprocessing()

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

def use_traned_model():
    # df = pd.read_csv(train_path)
    # loaded_model = pickle.load(open('E:\code\ml-projects\TabularWizard\results\trained_models\classification\creatures_xgbost_20231118_201353\finalized_lgbm_model.sav', 'rb'))
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
    print(len(train_data.index))
    print(len(train_data.columns))


    # results = []

    classifier = MLPNetClassifier(train_df = train_data, target_column = 'type')
    classifier.tune_hyper_parameters(scoring='accuracy')
    model = classifier.train()
    pickle.dump(model, open(SAVED_MODEL_FILE, 'wb'))

    evaluate = Evaluate()
    evaluations = evaluate.evaluate_train_and_test(model, classifier)
    
    print(f"model evaluations: {evaluations}")
    with open(SAVED_MODEL_EVALUATION, 'w') as file:
        file.write(evaluations)

    classifier.visualize_weights(model_folder=SAVED_MODEL_FOLDER)


if __name__ == '__main__':
    train_model()
    use_traned_model()