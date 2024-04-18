import pandas as pd
import json

def create_encoding_rules(df, threshold=0.05):
    encoding_rules = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        value_counts = df[col].value_counts(normalize=True)
        frequent_categories = value_counts[value_counts >= threshold].index.tolist()
        encoding_rules[col] = frequent_categories

    # # Save encoding rules to a JSON file
    # with open('encoding_rules.json', 'w') as file:
    #     json.dump(encoding_rules, file)
    
    return encoding_rules

def apply_encoding_rules(df, encoding_rules):
    df_encoded = df.copy()
    
    for col, rules in encoding_rules.items():
        # Apply 'Other' category to infrequent values
        df_encoded[col] = df[col].apply(lambda x: x if x in rules else 'Other')

        # Create a new DataFrame with the correct columns for one-hot encoding
        encoded_features = pd.get_dummies(df_encoded[col], prefix=col)
        # Ensure all columns that were created during training are present, initialized to False
        all_categories = {f"{col}_{category}": False for category in rules + ['Other']}
        for c in encoded_features.columns:
            all_categories[c] = encoded_features[c].astype(bool)
        encoded_df = pd.DataFrame(all_categories, index=df_encoded.index)

        # Drop the original column and append the new encoded columns
        df_encoded.drop(columns=[col], inplace=True)
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    # Correcting the data types
    for col in df_encoded.columns:
        if 'category' in col:
            df_encoded[col] = df_encoded[col].astype(bool)

    return df_encoded


# Example training DataFrame
df_train = pd.DataFrame({
    'category1': ['A', 'B', 'A', 'C', 'C', 'C', 'D', 'E', 'F', 'G', 'A', 'A', 'B', 'C'],
    'category2': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Y', 'Z', 'X', 'Z', 'Y', 'X'],
    'numeric1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
})

# Create and save encoding rules during training
encoding_rules = create_encoding_rules(df_train)
df_train_encoded = apply_encoding_rules(df_train, encoding_rules)

# Example inference DataFrame with different values
df_inference = pd.DataFrame({
    'category1': ['A', 'B', 'C', 'C', 'D', 'E', 'H'],  # 'H' is new
    'category2': ['X', 'Z', 'Z', 'Y', 'T', 'Y', 'X'],  # 'T' is new
    'numeric1': [15, 16, 17, 18, 19, 20, 21]
})

# # Load encoding rules
# with open('encoding_rules.json', 'r') as file:
#     encoding_rules = json.load(file)

# Apply encoding rules to inference data
df_inference_encoded = apply_encoding_rules(df_inference, encoding_rules)
print(df_inference_encoded)
