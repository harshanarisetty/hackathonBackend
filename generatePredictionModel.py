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
for col in ['worker', 'position', 'time', 'month']:
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

print(X_test)

# Save the trained model to a pickle file
with open('prediction.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
# Evaluate the model
# print(classification_report(y_test, y_pred))
