import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Generate a dataset of even and odd numbers
X = np.arange(1, 1001).reshape(-1, 1)
y = np.where(X % 2 == 0, 'even', 'odd')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model to a pickle file
with open('even_odd_classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Test the model
test_number = 15
predicted_class = model.predict([[test_number]])
print(f"{test_number} is predicted as {predicted_class[0]}")
