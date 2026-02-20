import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_and_preprocess_data(codebook_path, heart_failure_path):
    codebook = pd.read_csv(codebook_path)
    heart_failure = pd.read_csv(heart_failure_path)
    heart_failure['sex'] = heart_failure['sex'].map({1: 'M', 0: 'F'})

    return codebook, heart_failure

def prepare_knn_data(codebook, heart_failure, features_codebook, features_heart_failure):
    X_codebook = codebook[features_codebook]
    X_heart_failure = heart_failure[features_heart_failure]

    X_codebook_encoded = pd.get_dummies(X_codebook, drop_first=True)
    X_heart_failure_encoded = pd.get_dummies(X_heart_failure, drop_first=True)

    X_heart_failure_encoded = X_heart_failure_encoded.reindex(columns=X_codebook_encoded.columns, fill_value=0)

    scaler = StandardScaler()
    X_codebook_scaled = scaler.fit_transform(X_codebook_encoded)
    X_heart_failure_scaled = scaler.transform(X_heart_failure_encoded)

    return X_codebook_scaled, X_heart_failure_scaled

def find_similar_records(codebook, heart_failure, X_codebook_scaled, X_heart_failure_scaled, n_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(X_codebook_scaled)

    distances, indices = nbrs.kneighbors(X_heart_failure_scaled)

    merged_records = []
    for i, index_list in enumerate(indices):
        similar_records = codebook.iloc[index_list]
        for index in index_list:
            merged_record = {
                **similar_records.loc[index].to_dict(),
                **heart_failure.iloc[i].to_dict()
            }
            merged_records.append(merged_record)

    return pd.DataFrame(merged_records)

def train_random_forest(merged_data, features_for_prediction, target_column):
    X = merged_data[features_for_prediction]
    y = merged_data[target_column]

    X_encoded = pd.get_dummies(X, drop_first=True)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(model, X_encoded, y, cv=5, scoring='neg_mean_squared_error')
    cv_mse_scores = -cv_scores

    print("Cross-Validation MSE Scores:", cv_mse_scores)
    print("Mean Cross-Validation MSE:", cv_mse_scores.mean())
    print("Standard Deviation of Cross-Validation MSE:", cv_mse_scores.std())

    model.fit(X_encoded, y)

    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return model, feature_importance_df

def plot_feature_importance(feature_importance_df):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Regressor - Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

def main():
    codebook_path = 'codebook.csv' # Path to codebook.csv
    heart_failure_path = 'cleaned_file.csv' # Path to the heart_failure_dataset output file generated after isolation forest
    output_path = 'merged_dataset_knn.csv' # Path to the o/p file

    codebook, heart_failure = load_and_preprocess_data(codebook_path, heart_failure_path)

    features_codebook = ['Age', 'Sex']
    features_heart_failure = ['age', 'sex']
    X_codebook_scaled, X_heart_failure_scaled = prepare_knn_data(codebook, heart_failure, features_codebook, features_heart_failure)

    merged_data = find_similar_records(codebook, heart_failure, X_codebook_scaled, X_heart_failure_scaled)

    features_for_prediction = ['Age', 'Sex', 'HR', 'ejection_fraction', 'platelets',
                'serum_creatinine', 'serum_sodium', 'high_blood_pressure', 
                'diabetes', 'anaemia', 'smoking', 'ejection_fraction']
    target_column = 'RVEF'
    model, feature_importance_df = train_random_forest(merged_data, features_for_prediction, target_column)

    plot_feature_importance(feature_importance_df)

    merged_data.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
