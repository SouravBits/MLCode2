from flask import Flask, request, jsonify
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

app = Flask(__name__)

# Load the trained model and scaler
knn = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Set the tracking URI to the custom MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5001")

@app.route('/')
def home():
    return "Iris Classifier API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)

    # Standardize the input features
    features = scaler.transform(features)

    # Make a prediction
    prediction = knn.predict(features)
    predicted_class = prediction[0]

    # Map the prediction to the Iris target names
    target_names = ['setosa', 'versicolor', 'virginica']
    response = {
        'prediction': target_names[predicted_class]
    }

    # Log the prediction details to MLflow
    with mlflow.start_run():
        mlflow.log_param("input_features", data['features'])
        mlflow.log_metric("predicted_class", predicted_class)
        mlflow.sklearn.log_model(knn, "model", input_example=np.array(data['features']).reshape(1, -1))
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
