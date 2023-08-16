import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load the dataset
data = pd.read_csv('data.csv')

# Encode categorical variables
label_encoders = {}
categorical_cols = ['worker', 'position', 'time', 'month', 'activity']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split the data
X = data.drop('activity', axis=1)
y = data['activity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

with open('prediction2.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example input for prediction
input_worker = "Alicia Taylor"
input_position = "Compensation Administrator"
input_time = "Afternoon"
input_month = "Janfirsthalf"

# Convert text inputs to numerical values using label encoders
input_worker_encoded = label_encoders['worker'].transform([input_worker])[0]
input_position_encoded = label_encoders['position'].transform([input_position])[0]
input_time_encoded = label_encoders['time'].transform([input_time])[0]
input_month_encoded = label_encoders['month'].transform([input_month])[0]

# Create an input array for prediction
input_data = [[input_worker_encoded, input_position_encoded, input_time_encoded, input_month_encoded]]

# Make prediction using the trained model
predicted_activity = model.predict(input_data)

# Convert numerical prediction back to original activity text
predicted_activity_text = label_encoders['activity'].inverse_transform(predicted_activity)

print("\nPredicted Activity:", predicted_activity_text[0])
