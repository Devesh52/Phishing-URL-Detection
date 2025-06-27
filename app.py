from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import BertTokenizer
import numpy as np

app = Flask(__name__)

# Load the CNN model
model = load_model('221IT022_CNN_Model.h5')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Global storage for input-output
data_storage = []  # Stores list of (URL, Prediction) tuples

def tokenize(url):
    return tokenizer(url, padding="max_length", max_length=256, truncation=True, return_tensors="tf")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enter_input', methods=['POST'])
def enter_input():
    user_input = request.form['user_input']
    data_storage.append((user_input, ""))  # Store input with an empty prediction initially
    return "Input received successfully!"

@app.route('/display_output')
def display_output():
    if not data_storage:
        return "No input received yet."
    return f"Stored Output: {data_storage[-1][0]}"  # Display last stored input

@app.route('/store_output')
def store_output():
    if not data_storage:
        return "No data available to store!"

    with open("output.txt", "w") as file:
        for url, prediction in data_storage:
            file.write(f"URL: {url}\nPrediction: {prediction}\n\n")

    return "Output stored successfully in output.txt!"

@app.route('/predict', methods=['POST'])
def predict():
    url = request.json['url']

    # Tokenize the URL
    tokenized_url = tokenize(url)
    input_ids = np.array(tokenized_url['input_ids'])  # Extract input_ids
    input_ids = np.expand_dims(input_ids, axis=-1)    # Expand dims for CNN model

    # Make prediction
    prediction = model.predict(input_ids)[0][0]  # Extract single value

    # Convert prediction to label
    result = "Phishing" if prediction > 0.5 else "Benign"

    # Store result in global list
    data_storage.append((url, result))

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
