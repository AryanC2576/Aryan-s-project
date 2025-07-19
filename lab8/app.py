import pickle
import numpy as np
from flask import Flask, jsonify, render_template, request

# Load trained model
with open('house_data.pkl', 'rb') as f:
    model = pickle.load(f)

# Create Flask application
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if 'area' exists in form data
        if 'area' not in request.form:
            return jsonify({'error': 'Missing area input'}), 400

        # Get user input and convert to float
        area = float(request.form['area'])

        # Predict house price (fix: extract scalar from NumPy array)
        prediction = model.predict(np.array([[area]]))  # Returns a NumPy array
        predicted_price = float(prediction[0])  # Convert to float

        # Return prediction as JSON
        return jsonify({'predicted_price': round(predicted_price, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)

