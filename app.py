from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('RandomForest.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = {key: float(value) for key, value in request.form.items()}
    
    # Convert data to numpy array
    data_array = np.array(list(data.values())).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(data_array)
    
    # Return the prediction as JSON response
    return render_template('index.html', prediction_text='Predicted Class: {}'.format(prediction[0]), **data)

if __name__ == '__main__':
    app.run(debug=True)
