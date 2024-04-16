import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder

# Sample function to select encoding strategy based on cardinality
def encode_features(df, target_column, threshold=10):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    high_cardinality_cols = [col for col in categorical_columns if df[col].nunique() > threshold]
    low_cardinality_cols = [col for col in categorical_columns if col not in high_cardinality_cols]
    
    transformers = []
    if low_cardinality_cols:
        transformers.append(('onehot', OneHotEncoder(), low_cardinality_cols))
    if high_cardinality_cols:
        transformers.append(('target', TargetEncoder(), high_cardinality_cols))
    
    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    X_transformed = preprocessor.fit_transform(df.drop(target_column, axis=1), df[target_column])
    return pd.DataFrame(X_transformed)

# Example usage
df = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'baz'] * 10,
    'B': ['one', 'two', 'three', 'four', 'five', 'six', 'seven','eight', 'nine','ten',
           'one2', 'two2', 'three2', 'four2', 'five2', 'six2', 'seven2','eight2', 'nine2', 'ten2',
           'one3', 'two3', 'three3', 'four3', 'five3', 'six3', 'seven3','eight3', 'nine3','ten3', 
           'one4', 'two4', 'three4', 'four4', 'five4', 'six4', 'seven4','eight4', 'nine4', 'ten4' ] ,
    'target': [1, 0, 1, 0] * 10
})
encoded_df = encode_features(df, 'target')
print(encoded_df)