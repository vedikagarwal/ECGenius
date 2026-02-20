import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay


class DataPreprocessor:
    def __init__(self, file_path, rvef_threshold=50.0):
        self.file_path = file_path
        self.rvef_threshold = rvef_threshold
        self.data = pd.read_csv(file_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def encode_features(self):
        categorical_cols = ['PatientGroup', 'Sex', 'Split']
        self.label_encoders = {col: LabelEncoder() for col in categorical_cols}
        for col, encoder in self.label_encoders.items():
            self.data[col] = encoder.fit_transform(self.data[col])

    def create_rvef_status(self):
        self.data['RVEF_Status'] = self.data['RVEF'].apply(
            lambda x: 'normal' if x >= self.rvef_threshold else 'reduced'
        )

    def split_features_targets(self):
        X = self.data.drop(columns=['FileName', 'PatientHash', 'UltrasoundSystem', 'FPS', 'NumFrames', 
                                    'VideoViewType', 'VideoOrientation', 'VideoQuality', 'RVEDV', 
                                    'RVESV', 'RVEF', 'RVEF_Status', 'Split'])
        y_regression = self.data['RVEF']
        y_classification = self.data['RVEF_Status']
        return X, y_regression, y_classification

    def preprocess_data(self):
        self.create_rvef_status()
        self.encode_features()
        X, y_regression, y_classification = self.split_features_targets()
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_regression, test_size=0.2, random_state=42)
        X_train_class, X_test_class, y_class_train, y_class_test = train_test_split(X, y_classification, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        X_train_class = self.scaler.fit_transform(X_train_class)
        X_test_class = self.scaler.transform(X_test_class)
        return X_train, X_test, y_reg_train, y_reg_test, X_train_class, X_test_class, y_class_train, y_class_test


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.regressor = RandomForestRegressor(random_state=self.random_state)
        self.classifier = RandomForestClassifier(random_state=self.random_state)

    def train_regressor(self, X_train, y_train):
        self.regressor.fit(X_train, y_train)

    def train_classifier(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict_regressor(self, X_test):
        return self.regressor.predict(X_test)

    def predict_classifier(self, X_test):
        return self.classifier.predict(X_test)


class ModelEvaluator:
    def __init__(self, regressor, classifier):
        self.regressor = regressor
        self.classifier = classifier

    def evaluate_regression(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print("Regression Results:")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        return mse, mae

    def evaluate_classification(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        classification_rep = classification_report(y_true, y_pred, output_dict=True)
        print("\nClassification Results:")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        return accuracy, classification_rep

    def plot_evaluation_metrics(self, mse, mae, accuracy, classification_rep):
        regression_metrics = ['MSE', 'MAE']
        values_regression = [mse, mae]

        precision = classification_rep['weighted avg']['precision']
        recall = classification_rep['weighted avg']['recall']
        f1_score = classification_rep['weighted avg']['f1-score']

        classification_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values_classification = [accuracy, precision, recall, f1_score]

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.barplot(x=regression_metrics, y=values_regression, ax=ax[0], palette='Blues')
        ax[0].set_title('Regression Evaluation Metrics')
        ax[0].set_ylabel('Error Value')

        sns.barplot(x=classification_metrics, y=values_classification, ax=ax[1], palette='Greens')
        ax[1].set_title('Classification Evaluation Metrics')
        ax[1].set_ylabel('Score')
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, X_columns):
        feature_importances = self.regressor.feature_importances_
        importance_df = pd.DataFrame({'Feature': X_columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title('Feature Importance - Random Forest Regressor')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()

    def plot_partial_dependence(self, X_train, feature_names):
        plt.figure(figsize=(12, 8))
        PartialDependenceDisplay.from_estimator(self.regressor, X_train, features=[0, 1, 2, 3], feature_names=feature_names, grid_resolution=50)
        plt.suptitle('Partial Dependence Plots (PDP) for Features')
        plt.tight_layout()
        plt.show()



file_path = 'codebook.csv'  # Enter the path for your codebook.csv file
data_processor = DataPreprocessor(file_path)
X_train, X_test, y_reg_train, y_reg_test, X_train_class, X_test_class, y_class_train, y_class_test = data_processor.preprocess_data()

trainer = ModelTrainer()
trainer.train_regressor(X_train, y_reg_train)
trainer.train_classifier(X_train_class, y_class_train)

y_reg_pred = trainer.predict_regressor(X_test)
y_class_pred = trainer.predict_classifier(X_test_class)

evaluator = ModelEvaluator(trainer.regressor, trainer.classifier)
mse, mae = evaluator.evaluate_regression(y_reg_test, y_reg_pred)
accuracy, classification_rep = evaluator.evaluate_classification(y_class_test, y_class_pred)
evaluator.plot_evaluation_metrics(mse, mae, accuracy, classification_rep)
evaluator.plot_feature_importance(data_processor.data.drop(columns=['FileName', 'PatientHash','UltrasoundSystem', 'FPS', 'NumFrames', 'VideoViewType', 'VideoOrientation','VideoQuality','RVEDV', 'RVESV', 'RVEF', 'RVEF_Status','Split']).columns)
evaluator.plot_partial_dependence(X_train, data_processor.data.columns)
