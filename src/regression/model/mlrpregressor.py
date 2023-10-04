import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

def automatic_nn(dataframe, target_column):
    # Separate features and target variable
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    # Define numerical and categorical feature columns
    num_features = X.select_dtypes(exclude=['object']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    # Create transformers
    num_transformer = Pipeline(steps=[
        ('num_imputer', SimpleImputer(strategy='mean')),  # Impute NaNs with mean value
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),  # Impute NaNs with most frequent category
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a column transformer that will allow us to manipulate the data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Setting up neural network parameters
    # Just as an example, setting hidden_layer_sizes dynamically based on number of features
    n_cols = X_train.shape[1]
    hidden_layer_sizes = (int(np.sqrt(n_cols + 1)),)
    
    # Create and train the neural network model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('nn_regressor', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, 
            max_iter=100000,  # Increase max_iter if model doesn't converge
            random_state=42,
            solver='adam',  # or 'lbfgs',
            alpha=0.5  # L2 regularization term; adjust as needed
            # early_stopping=True,
            # validation_fraction=0.1,  # Fraction of training data to set aside as validation set for early stopping
            # n_iter_no_change=10
        ))
    ])

    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print("Model Test performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print("R2 Score:", r2)

    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, predictions)
    
    print("Model Train performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print("R2 Score:", r2)

    return model

# Example usage:
# Assuming df is your dataframe and 'target' is your target variable
# model = automatic_nn(df, 'target')
