import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data(codebook_file, heart_failure_file):
    codebook_df = pd.read_csv(codebook_file)
    heart_failure_df = pd.read_csv(heart_failure_file)
    return codebook_df, heart_failure_df


def preprocess_data(codebook_df, heart_failure_df):
    heart_failure_df['sex'] = heart_failure_df['sex'].map({1: 'M', 0: 'F'})
    codebook_df['Age'] = codebook_df['Age'].round().astype(int)
    heart_failure_df['age'] = heart_failure_df['age'].astype(int)
    heart_failure_df = heart_failure_df.groupby(['age', 'sex'], as_index=False).mean()
    return codebook_df, heart_failure_df


def merge_data(codebook_df, heart_failure_df):
    merged_data = pd.merge(codebook_df, heart_failure_df, 
                           left_on=['Age', 'Sex'], 
                           right_on=['age', 'sex'], 
                           how='left')
    assert len(merged_data) == len(codebook_df), "The number of rows in the merged DataFrame does not match the original codebook DataFrame."
    return merged_data


def impute_missing_values(data, features_to_impute):
    imputation_data = data[['Age', 'Sex'] + features_to_impute].copy()
    imputation_data['Sex'] = imputation_data['Sex'].map({'M': 1, 'F': 0})
    columns_with_all_nan = imputation_data.columns[imputation_data.isnull().all()]
    imputation_data.drop(columns_with_all_nan, axis=1, inplace=True)

    imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
    imputed_data = imputer.fit_transform(imputation_data)
    imputed_df = pd.DataFrame(imputed_data, columns=imputation_data.columns).round(2)
    imputed_df['Sex'] = imputed_df['Sex'].map({1: 'M', 0: 'F'})

    for feature in features_to_impute:
        if feature in imputed_df.columns:
            data[feature] = imputed_df[feature]

    return data


def train_model(data, features, target_column):
    X = data[features]
    y = data[target_column]
    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = ((y_test - y_pred) ** 2).mean()
    return model, mse, X.columns


def plot_feature_importance(model, feature_names):
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Regressor - Feature Importance')
    plt.show()


def save_merged_data(data, output_file):
    data.fillna(-1, inplace=True)
    data.drop(columns=['age', 'sex'], inplace=True)
    data.to_csv(output_file, index=False)


def main():
    codebook_file = 'codebook.csv'  # Path to the codebook.csv file
    heart_failure_file = 'cleaned_file.csv' # Path to the heart_failure_dataset output file generated after isolation forest
    output_file = 'merged_codebook.csv'
    
    features_to_impute = ['anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                          'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
                          'smoking', 'DEATH_EVENT']
    features = ['Age', 'Sex', 'HR', 'ejection_fraction', 'platelets',
                'serum_creatinine', 'serum_sodium', 'high_blood_pressure', 
                'diabetes', 'anaemia', 'smoking', 'ejection_fraction']
    target_column = 'RVEF'

    codebook_df, heart_failure_df = load_data(codebook_file, heart_failure_file)
    codebook_df, heart_failure_df = preprocess_data(codebook_df, heart_failure_df)
    merged_data = merge_data(codebook_df, heart_failure_df)
    merged_data = impute_missing_values(merged_data, features_to_impute)

    print(f"Number of rows in codebook_df before merging: {len(codebook_df)}")
    print(f"Number of rows in merged_data after merging: {len(merged_data)}")

    model, mse, feature_names = train_model(merged_data, features, target_column)
    print("Mean Squared Error:", mse)

    plot_feature_importance(model, feature_names)
    save_merged_data(merged_data, output_file)


if __name__ == "__main__":
    main()
