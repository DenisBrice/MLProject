from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)
model = load('SVC_model.joblib')

@app.route('/', methods=['GET'])
def home():
    return "Welcome to my prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Check if data is None (no data was provided)
    if data is None:
        return jsonify({'error': 'No data provided'}), 400

    # Check if data is a list (as expected)
    if not isinstance(data, list):
        return jsonify({'error': 'Data should be a list of lists'}), 400

    # Check if all elements in data are lists (as expected)
    if not all(isinstance(i, list) for i in data):
        return jsonify({'error': 'Each item in data should be a list'}), 400

    try:
        predictions = model.predict(data)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    

