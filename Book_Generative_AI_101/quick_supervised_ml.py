from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

iris_data = load_iris()

# Split the dataset into features and targets
X = iris_data.data
y = iris_data.target

# Perform a train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape: ", X_train.shape)
print("Test data shape: ", X_test.shape)

# Create a SVM classifier with a linear kernel
clf = svm.SVC(kernel='linear')

# Train the classifier with the training data
clf.fit(X_train, y_train)

# Use the trained classifier to make predictions with the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy*100}%")