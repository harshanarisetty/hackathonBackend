import pickle
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

# Encode categorical variables
label_encoders = {}
categorical_cols = ['worker', 'position', 'time', 'month', 'activity']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

app = Flask(__name__)

data = pd.read_csv('data.csv')



# Load the trained model from the pickle file
with open('even_odd_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the trained model from the pickle file
with open('prediction2.pkl', 'rb') as model_file2:
    model2 = pickle.load(model_file2)

@app.route('/api/check_even_odd', methods=['GET'])
def check_even_odd():
    try:
        number = int(request.args.get('number'))
        predicted_class = model.predict([[number]])[0]
        result = {'number': number, 'prediction': predicted_class}
        return jsonify(result), 200
    except ValueError:
        return jsonify({'error': 'Invalid input'}), 400

@app.route('/api/searchPrediction', methods=['GET'])
def getPrediction():
    try:
        input_worker = request.args.get('worker')
        input_position = request.args.get('position')
        input_time = request.args.get('time')
        input_month = request.args.get('month')

        # Convert text inputs to numerical values using label encoders
        input_worker_encoded = label_encoders['worker'].transform([input_worker])[0]
        input_position_encoded = label_encoders['position'].transform([input_position])[0]
        input_time_encoded = label_encoders['time'].transform([input_time])[0]
        input_month_encoded = label_encoders['month'].transform([input_month])[0]
        
        

        input_data = [[input_worker_encoded, input_position_encoded, input_time_encoded, input_month_encoded]]
        print(input_data)

        # predicted_activity = model2.predict(input_data)


        predicted_activity_many = model2.predict_proba(input_data)  # Get class probabilities

        # Choose the top N classes for each prediction
        top_n = 3  # Change this to the desired number of top predictions
        top_n_preds = [model2.classes_[probs.argsort()[-top_n:][::-1]] for probs in predicted_activity_many]

        # Get the corresponding activity labels for the top N predictions
        top_n_activity_labels = [label_encoders['activity'].inverse_transform(preds) for preds in top_n_preds]
        result = [list(preds) for preds in top_n_activity_labels]
        print( top_n_activity_labels)
        return jsonify(result[0]), 200
    
    except ValueError as e:
        print(e.args[0])
        return jsonify({'error': 'Invalid input'}), 400
if __name__ == '__main__':
    app.run(debug=True)
