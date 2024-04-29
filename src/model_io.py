import joblib
from sklearn.model_selection import train_test_split


def save_model(model, path):
    # bst.save_model (path)
    joblib.dump (model, path)


def load_model(path):
    return joblib.load(path)


def predict_with_saved_model( model, df, target_column):
    df = df.drop ([target_column], axis=1)
    df = df.select_dtypes (include=['number']).copy ()
    predictions = model.predict(df)
    return predictions