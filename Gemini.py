import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Load MNIST dataset
mnist = datasets.fetch_openml(name='mnist_784', version=1)

# Separate features (X) and labels (y)
X = mnist.data
y = mnist.target

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the MLP architecture
mlp_classifier = MLPClassifier(hidden_layer_sizes=(20, 10), random_state=42)

# Train the model
mlp_classifier.fit(X_train, y_train)

# Access validation scores (assuming early stopping is enabled)
validation_scores = mlp_classifier.validation_scores_  # Use plural form

# Conditional plotting to prevent errors if early stopping is not used
if validation_scores is not None:
    plt.plot(validation_scores)
    plt.title('Validation Scores')
    plt.show()

# Make predictions on the test set
y_pred = mlp_classifier.predict(X_test)

# Generate classification report
print(classification_report(y_test, y_pred))
