import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd
from typing import Dict, Any, Union

# Deep Learning Model Definition


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class RecommenderModel:
    def __init__(self, model_params: Dict[str, Any] = None, model_type: str = 'neural_network'):
        """
        Initialize the RecommenderModel class.

        Args:
            model_params (Dict[str, Any], optional): Parameters for the model. Defaults to None.
            model_type (str): Type of model to use ('neural_network' or 'random_forest').
        """
        self.model_params = model_params if model_params else {}
        self.model_type = model_type
        self.model = None

    def build_model(self):
        """
        Build the recommendation model using the specified parameters.
        """
        try:
            if self.model_type == 'random_forest':
                self.model = RandomForestRegressor(**self.model_params)
                print(
                    f"RandomForest model built with parameters: {self.model_params}")

            elif self.model_type == 'neural_network':
                input_size = self.model_params.get('input_size', 10)
                hidden_size = self.model_params.get('hidden_size', 64)
                output_size = self.model_params.get('output_size', 1)
                self.model = SimpleNN(input_size, hidden_size, output_size)
                print(
                    f"Neural Network model built with parameters: {self.model_params}")

            else:
                raise ValueError(
                    "Unsupported model type. Use 'random_forest' or 'neural_network'.")
        except Exception as e:
            print(f"Error building model: {e}")
            self.model = None

    def train_model(self, data: pd.DataFrame, target: str):
        """
        Train the model on the provided data.

        Args:
            data (pd.DataFrame): The input data to train on.
            target (str): The name of the target column.
        """
        try:
            if self.model_type == 'random_forest':
                X = data.drop(columns=[target])
                y = data[target]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)

                with mlflow.start_run():
                    self.model.fit(X_train, y_train)

                    # Log the model parameters
                    mlflow.log_params(self.model_params)

                    # Evaluate and log metrics
                    train_preds = self.model.predict(X_train)
                    test_preds = self.model.predict(X_test)
                    train_mse = mean_squared_error(y_train, train_preds)
                    test_mse = mean_squared_error(y_test, test_preds)

                    mlflow.log_metric("train_mse", train_mse)
                    mlflow.log_metric("test_mse", test_mse)
                    mlflow.log_metric(
                        "train_r2", r2_score(y_train, train_preds))
                    mlflow.log_metric("test_r2", r2_score(y_test, test_preds))

                    # Log the model
                    mlflow.sklearn.log_model(self.model, "random_forest_model")

                    print(
                        f"RandomForest Model trained successfully. Train MSE: {train_mse}, Test MSE: {test_mse}")

            elif self.model_type == 'neural_network':
                X = data.drop(columns=[target]).values
                y = data[target].values

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(
                    y_train, dtype=torch.float32).view(-1, 1)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_test_tensor = torch.tensor(
                    y_test, dtype=torch.float32).view(-1, 1)

                criterion = nn.MSELoss()
                optimizer = optim.Adam(self.model.parameters(), lr=0.001)

                with mlflow.start_run():
                    # Training loop
                    num_epochs = self.model_params.get('num_epochs', 100)
                    for epoch in range(num_epochs):
                        self.model.train()
                        optimizer.zero_grad()
                        outputs = self.model(X_train_tensor)
                        loss = criterion(outputs, y_train_tensor)
                        loss.backward()
                        optimizer.step()
                        if (epoch + 1) % 10 == 0:
                            print(
                                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

                    # Evaluate the model
                    self.model.eval()
                    with torch.no_grad():
                        train_preds = self.model(X_train_tensor).numpy()
                        test_preds = self.model(X_test_tensor).numpy()

                    train_mse = mean_squared_error(y_train, train_preds)
                    test_mse = mean_squared_error(y_test, test_preds)

                    mlflow.log_metric("train_mse", train_mse)
                    mlflow.log_metric("test_mse", test_mse)
                    mlflow.log_metric(
                        "train_r2", r2_score(y_train, train_preds))
                    mlflow.log_metric("test_r2", r2_score(y_test, test_preds))

                    # Log the model
                    mlflow.pytorch.log_model(
                        self.model, "neural_network_model")

                    print(
                        f"Neural Network Model trained successfully. Train MSE: {train_mse}, Test MSE: {test_mse}")
            else:
                raise ValueError(
                    "Unsupported model type for training. Use 'random_forest' or 'neural_network'.")
        except Exception as e:
            print(f"Error training model: {e}")

    def evaluate_model(self, data: pd.DataFrame, target: str) -> Dict[str, float]:
        """
        Evaluate the model on the provided data.

        Args:
            data (pd.DataFrame): The input data to evaluate on.
            target (str): The name of the target column.
        
        Returns:
            Dict[str, float]: Evaluation metrics such as MSE and R2 score.
        """
        try:
            if self.model_type == 'random_forest':
                X = data.drop(columns=[target])
                y = data[target]
                predictions = self.model.predict(X)
                mse = mean_squared_error(y, predictions)
                r2 = r2_score(y, predictions)
                print(f"Evaluation results - MSE: {mse}, R2 Score: {r2}")
                return {"mse": mse, "r2": r2}

            elif self.model_type == 'neural_network':
                X = data.drop(columns=[target]).values
                y = data[target].values
                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

                self.model.eval()
                with torch.no_grad():
                    predictions = self.model(X_tensor).numpy()

                mse = mean_squared_error(y, predictions)
                r2 = r2_score(y, predictions)
                print(f"Evaluation results - MSE: {mse}, R2 Score: {r2}")
                return {"mse": mse, "r2": r2}

            else:
                raise ValueError(
                    "Unsupported model type for evaluation. Use 'random_forest' or 'neural_network'.")
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {"mse": float('inf'), "r2": float('-inf')}

# Example usage:
# from models.recommender_model import RecommenderModel

# data = pd.DataFrame({
#     'feature1': [1, 2, 3, 4, 5],
#     'feature2': [10, 20, 30, 40, 50],
#     'target': [15, 25, 35, 45, 55]
# })

# For Neural Network
# model_params = {"input_size": 2, "hidden_size": 64, "output_size": 1, "num_epochs": 100}
# recommender = RecommenderModel(model_params=model_params, model_type='neural_network')
# recommender.build_model()
# recommender.train_model(data, 'target')
# evaluation_metrics = recommender.evaluate_model(data, 'target')
# print(evaluation_metrics)

# For Random Forest
# model_params = {"n_estimators": 100, "random_state": 42}
# recommender = RecommenderModel(model_params=model_params, model_type='random_forest')
# recommender.build_model()
# recommender.train_model(data, 'target')
# evaluation_metrics =
