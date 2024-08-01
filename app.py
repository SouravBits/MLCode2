from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
knn = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

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

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)