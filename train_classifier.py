import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the processed data from pickle file
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("The pickle file 'data.pickle' was not found. Please check the file path.")
    exit(1)

# Ensure all feature vectors have the same length
max_length = max(len(sample) for sample in data_dict['data'])
data = np.array([np.pad(sample, (0, max_length - len(sample)), mode='constant') if len(sample) < max_length else sample for sample in data_dict['data']])
labels = np.asarray(data_dict['labels'])

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForest classifier
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Predict the labels for the test set
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_test, y_predict)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
