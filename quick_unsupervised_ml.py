from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# (1) Load the Iris dataset
iris = datasets.load_iris()

# (2) Split the predictor and target feature
X = iris.data
y = iris.target

# We will standardize our features for better performance of k-means
scaler = StandardScaler()
X = scaler.fit_transform(X)

# (3) Perform a train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (4) Fit K-means clustering with 3 clusters on training data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)

print("This is X_test")
print(X_test)

# (5) Predict the cluster assignment on the test data
predictions = kmeans.predict(X_test)

# Print the predictions
print(predictions)

# save model
joblib.dump(kmeans, 'kmeans_model.pkl')