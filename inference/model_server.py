import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Any, Dict


class ModelServer:
    def __init__(self):
        self.model = None
        self.app = FastAPI()

    def load_model(self, model_path: str):
        """
        Load the model from the given path.

        Args:
            model_path (str): Path to the model to load.
        """
        try:
            self.model = mlflow.pyfunc.load_model(model_path)
            print(f"Model loaded from {model_path} successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")

    def serve_predictions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serve predictions for the provided data.

        Args:
            data (Dict[str, Any]): Data to predict on, structured as a dictionary.
        
        Returns:
            Dict[str, Any]: Predictions made by the model.
        """
        try:
            # Convert dictionary to DataFrame
            input_data = pd.DataFrame([data])
            print(f"Input data for prediction: {input_data}")

            # Make predictions
            predictions = self.model.predict(input_data)
            print(f"Predictions: {predictions}")

            # Return predictions in a dictionary format
            return {"predictions": predictions.tolist()}
        except Exception as e:
            print(f"Error serving predictions: {e}")
            raise HTTPException(
                status_code=500, detail="Prediction serving failed")

    def setup_routes(self):
        """
        Setup FastAPI routes for the model server.
        """
        @self.app.post("/predict", response_model=Dict[str, Any])
        async def predict(data: Dict[str, Any]):
            return self.serve_predictions(data)

    def run_server(self):
        """
        Run the FastAPI server to serve the model.
        """
        import uvicorn
        uvicorn.run(self.app, host='0.0.0.0', port=8000)

# Example usage
# if __name__ == "__main__":
#     server = ModelServer()
#     server.load_model("path/to/your/model")
#     server.setup_routes()
#     server.run_server()
