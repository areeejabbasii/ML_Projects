import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
class Model:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        print("Data loaded successfully.")

    def preprocess_data(self):
        # Only convert specific columns with k, m, b, % suffixes
        columns_to_convert = ['posts', 'followers', 'avg_likes', '60_day_eng_rate', 'new_post_avg_like', 'total_likes']
        
        def convert_to_numeric(value):
            if isinstance(value, str):
                value = value.strip()
                try:
                    if value.endswith('%'):
                        return float(value.rstrip('%')) / 100
                    elif value.endswith('b'):
                        return float(value.rstrip('b')) * 1e9
                    elif value.endswith('m'):
                        return float(value.rstrip('m')) * 1e6
                    elif value.endswith('k'):
                        return float(value.rstrip('k')) * 1e3
                    else:
                        return float(value)
                except:
                    return np.nan
            return value
        
        # Convert only numeric columns
        for col in columns_to_convert:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(convert_to_numeric)
        
        # Fill missing values
        for col in self.data.columns:
            if self.data[col].dtype in ['float64', 'int64']:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            else:
                self.data[col].fillna('Unknown', inplace=True)
        
        # Encode categorical variables
        self.data = pd.get_dummies(self.data, drop_first=True)
        print("Data preprocessed successfully.")

    def split_data(self, target_column):
        # Exclude specified columns
        exclude_columns = ['channel_info', 'influence_score', 'avg_likes', '60_day_eng_rate']
        columns_to_drop = [target_column] + exclude_columns
        X = self.data.drop(columns=columns_to_drop, errors='ignore')
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split into training and testing sets.")

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")
        print("Classification Report:")
        print(report)