import numpy as np
import gzip
import matplotlib.pyplot as plt

### Flattening and One-Hot Encoding ###
# One-hot encoding of the labels
def one_hot_encoding(label_data):
    encoded_labels = np.zeros((label_data.shape[0], 10))
    encoded_labels[np.arange(label_data.shape[0]), label_data] = 1
    return encoded_labels

# Function to read pixel data from the dataset
def read_pixels(data_path):
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), 'B', offset=16).astype('float32')
    normalized_pixels = pixel_data / 255
    flattened_pixels = normalized_pixels.reshape(-1, 28*28)
    return flattened_pixels

# Function to read label data from the dataset
def read_labels(data_path):
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), 'B', offset=8)
    one_hot_encoding_labels = one_hot_encoding(label_data)
    return one_hot_encoding_labels

# Function to read the entire dataset
def read_dataset():
    X_train = read_pixels("data/train-images-idx3-ubyte.gz")
    y_train = read_labels("data/train-labels-idx1-ubyte.gz")
    X_test = read_pixels("data/t10k-images-idx3-ubyte.gz")
    y_test = read_labels("data/t10k-labels-idx1-ubyte.gz")
    return X_train, y_train, X_test, y_test

# Read the dataset
X_train, y_train, X_test, y_test = read_dataset()

# Print the shapes of the data arrays
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

### Question 2.1 ###
# Function to initialize weights based on the specified method
def initialize_weights(input_size, output_size, method='normal'):
    if method == 'zeros':
        return np.zeros((input_size, output_size))
    elif method == 'uniform':
        return np.random.uniform(low=-1, high=1, size=(input_size, output_size))
    elif method == 'normal':
        return np.random.normal(loc=0, scale=1, size=(input_size, output_size))
    else:
        raise ValueError(f"Invalid weight initialization method: {method}")

# Softmax activation function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Cross-entropy loss with L2 regularization
def calculate_loss(y_true, y_pred, weights, lambda_reg):
    m = y_true.shape[0]
    cross_entropy_loss = -np.sum(y_true * np.log(y_pred))
    l2_regularization = (lambda_reg / (2 * m)) * np.sum(weights**2)
    total_loss = (cross_entropy_loss + l2_regularization) / m
    return total_loss

# Gradient descent update rule with L2 regularization
def update_weights(X, y_true, y_pred, weights, learning_rate, lambda_reg):
    m = X.shape[0]
    gradient = np.dot(X.T, (y_pred - y_true)) + (lambda_reg / m) * weights
    weights -= learning_rate * gradient
    return weights

# Logistic Regression training function
def train_logistic_regression(X_train, y_train, X_val, y_val, number_of_epochs, batch_size, learning_rate, lambda_reg, weight_init='normal'):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    weights = initialize_weights(input_size, output_size, method=weight_init)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(number_of_epochs):
        for batch_start in range(0, X_train.shape[0], batch_size):
            batch_end = batch_start + batch_size
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]

            # Forward pass
            z = np.dot(X_batch, weights)
            y_pred = softmax(z)

            # Calculate loss
            loss = calculate_loss(y_batch, y_pred, weights, lambda_reg)

            # Backward pass and weight update
            weights = update_weights(X_batch, y_batch, y_pred, weights, learning_rate, lambda_reg)

        # Evaluate on validation set
        val_z = np.dot(X_val, weights)
        val_y_pred = softmax(val_z)
        val_loss = calculate_loss(y_val, val_y_pred, weights, lambda_reg)

        # Calculate training accuracy
        train_predictions = predict(X_train, weights)
        train_accuracy = calculate_accuracy(np.argmax(y_train, axis=1), train_predictions)

        # Calculate validation accuracy
        val_predictions = predict(X_val, weights)
        val_accuracy = calculate_accuracy(np.argmax(y_val, axis=1), val_predictions)

        train_losses.append(loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{number_of_epochs} - Training Loss: {loss:.4f} - Validation Loss: {val_loss:.4f} - Training Accuracy: {train_accuracy:.4f} - Validation Accuracy: {val_accuracy:.4f}")

    return weights, train_losses, val_losses, train_accuracies, val_accuracies

# Function to make predictions using trained weights
def predict(X, weights):
    z = np.dot(X, weights)
    y_pred = softmax(z)
    return np.argmax(y_pred, axis=1)

# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = y_true.shape[0]
    accuracy = correct_predictions / total_samples
    return accuracy

# Function to calculate confusion matrix
def confusion_matrix(y_true, y_pred, number_of_classes):
    conf_matrix = np.zeros((number_of_classes, number_of_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        conf_matrix[true_label, pred_label] += 1
    return conf_matrix

# Read the dataset
X_train, y_train, X_test, y_test = read_dataset()

# Split the training set into training and validation sets
number_of_validation = 10000
X_val = X_train[:number_of_validation]
y_val = y_train[:number_of_validation]
X_train_partial = X_train[number_of_validation:]
y_train_partial = y_train[number_of_validation:]

# Set hyperparameters with the default values
number_of_epochs = 100
default_batch_size = 200
default_weight_init_method = 'normal'
default_learning_rate = 5e-4
default_lambda_reg = 1e-4

# Train the default Logistic Regression model
trained_weights, _, _, _, _ = train_logistic_regression(
    X_train_partial, y_train_partial, X_val, y_val, number_of_epochs, default_batch_size, default_learning_rate, default_lambda_reg, weight_init=default_weight_init_method
)

# Make predictions on the test set using the default model
test_predictions = predict(X_test, trained_weights)

# Calculate test accuracy for the default model
y_test_labels = np.argmax(y_test, axis=1)
test_accuracy = calculate_accuracy(y_test_labels, test_predictions)
print(f"Default Test Accuracy: {test_accuracy:.4f}")

# Calculate confusion matrix for the default model
number_of_classes = 10
conf_matrix = confusion_matrix(y_test_labels, test_predictions, number_of_classes)

# Display confusion matrix for the default model
print("\nDefault Confusion Matrix:")
print(conf_matrix)

### Question 2.2 ###
# Function to perform hyperparameter experiments for batch size
def perform_batch_size_experiments(X_train, y_train, X_val, y_val, number_of_epochs):
    batch_sizes = [1, 64, 50000]

    print("Experiment 1: Varying Batch Size")

    for batch_size in batch_sizes:
        print("Training with " + str(batch_size) + ":")
        # Train the Logistic Regression model with modified batch size
        _, _, _, _, val_accuracies = train_logistic_regression(
            X_train, y_train, X_val, y_val, number_of_epochs, batch_size, learning_rate, lambda_reg)

        # Plot epoch vs accuracy
        plt.plot(range(1, number_of_epochs + 1), val_accuracies, label=f'Batch Size: {batch_size}')

        print("\nTraining with " + str(batch_size) + " done.\n")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Batch Size Experiments')
    plt.show()

# Function to perform hyperparameter experiments for weight initialization
def perform_weight_init_experiments(X_train, y_train, X_val, y_val, number_of_epochs):
    weight_init_methods = ['zeros', 'uniform', 'normal']

    print("Experiment 2: Varying Weight Initialization Method")

    for weight_init_method in weight_init_methods:
        print("Training with " + weight_init_method + ":")
        # Train the Logistic Regression model with modified weight initialization
        _, _, _, _, val_accuracies = train_logistic_regression(
            X_train, y_train, X_val, y_val, number_of_epochs, batch_size, learning_rate, lambda_reg, weight_init=weight_init_method)

        # Plot epoch vs accuracy
        plt.plot(range(1, number_of_epochs + 1), val_accuracies, label=f'Weight Init: {weight_init_method}')

        print("\nTraining with " + weight_init_method + " done.\n")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Weight Initialization Experiments')
    plt.show()

# Function to perform hyperparameter experiments for learning rate
def perform_learning_rate_experiments(X_train, y_train, X_val, y_val, number_of_epochs):
    learning_rates = [0.1, 1e-3, 1e-4, 1e-5]

    print("Experiment 3: Varying Learning Rate")

    for learning_rate in learning_rates:
        print("Training with " + str(learning_rate) + ":")
        # Train the Logistic Regression model with modified learning rate
        _, _, _, _, val_accuracies = train_logistic_regression(
            X_train, y_train, X_val, y_val, number_of_epochs, batch_size, learning_rate, lambda_reg)

        # Plot epoch vs accuracy
        plt.plot(range(1, number_of_epochs + 1), val_accuracies, label=f'Learning Rate: {learning_rate}')

        print("\nTraining with " + str(learning_rate) + " done.\n")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Learning Rate Experiments')
    plt.show()

# Function to perform hyperparameter experiments for regularization coefficient
def perform_lambda_reg_experiments(X_train, y_train, X_val, y_val, number_of_epochs):
    lambda_regs = [1e-2, 1e-4, 1e-9]

    print("Experiment 4: Varying Regularization Coefficient")

    for lambda_reg in lambda_regs:
        print("Training with " + str(lambda_reg) + ":")
        # Train the Logistic Regression model with modified regularization coefficient
        _, _, _, _, val_accuracies = train_logistic_regression(
            X_train, y_train, X_val, y_val, number_of_epochs, batch_size, learning_rate, lambda_reg)

        # Plot epoch vs accuracy
        plt.plot(range(1, number_of_epochs + 1), val_accuracies, label=f'Regularization Coefficient: {lambda_reg}')

        print("\nTraining with " + str(lambda_reg) + " done.\n")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Regularization Coefficient Experiments')
    plt.show()

# Read the dataset
X_train, y_train, X_test, y_test = read_dataset()

# Split the training set into training and validation sets
number_of_validation = 10000
X_val = X_train[:number_of_validation]
y_val = y_train[:number_of_validation]
X_train_partial = X_train[number_of_validation:]
y_train_partial = y_train[number_of_validation:]

# Set default hyperparameters
number_of_epochs = 100
batch_size = 200
learning_rate = 5e-4
lambda_reg = 1e-4

# Perform hyperparameter experiments
perform_batch_size_experiments(X_train_partial, y_train_partial, X_val, y_val, number_of_epochs)
perform_weight_init_experiments(X_train_partial, y_train_partial, X_val, y_val, number_of_epochs)
perform_learning_rate_experiments(X_train_partial, y_train_partial, X_val, y_val, number_of_epochs)
perform_lambda_reg_experiments(X_train_partial, y_train_partial, X_val, y_val, number_of_epochs)

### Question 2.3 ###
# Set hyperparameters with the best values
best_batch_size = 200
best_weight_init_method = 'zeros'
best_learning_rate = 0.001
best_lambda_reg = 0.0001

# Train the optimal Logistic Regression model
optimal_trained_weights, _, _, _, _ = train_logistic_regression(
    X_train_partial, y_train_partial, X_val, y_val, number_of_epochs, best_batch_size, best_learning_rate, best_lambda_reg, weight_init=best_weight_init_method
)

# Make predictions on the test set using the optimal model
optimal_test_predictions = predict(X_test, optimal_trained_weights)

# Calculate test accuracy for the optimal model
optimal_test_accuracy = calculate_accuracy(y_test_labels, optimal_test_predictions)
print(f"Optimal Test Accuracy: {optimal_test_accuracy:.4f}")

# Calculate confusion matrix for the optimal model
optimal_conf_matrix = confusion_matrix(y_test_labels, optimal_test_predictions, number_of_classes)

# Display confusion matrix for the optimal model
print("\nOptimal Confusion Matrix:")
print(optimal_conf_matrix)

### Question 2.4 ###
# Plot the weight vectors as images
plt.figure(figsize=(12, 8))
for i, weight_vector in enumerate(optimal_trained_weights.T):
    # Reshape the weight into a 28x28 image
    weight_image = weight_vector.reshape(28, 28)

    # Plot the image
    plt.subplot(2, 5, i + 1) 
    plt.imshow(weight_image, cmap=plt.cm.gray, vmin=0.5 * weight_image.min(), vmax=0.5 * weight_image.max())
    plt.title(f'Weight Vector {i + 1}')

plt.show()

### Question 2.5 ###
def calculate_metrics(conf_matrix):
    number_of_classes = conf_matrix.shape[0]

    precision = np.zeros(number_of_classes)
    recall = np.zeros(number_of_classes)
    f1_score = np.zeros(number_of_classes)
    f2_score = np.zeros(number_of_classes)

    for i in range(number_of_classes):
        true_positive = conf_matrix[i, i]
        false_positive = np.sum(conf_matrix[:, i]) - true_positive
        false_negative = np.sum(conf_matrix[i, :]) - true_positive

        # Precision
        precision[i] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

        # Recall
        recall[i] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        # F1 Score
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

        # F2 Score
        beta = 2
        f2_score[i] = (1 + beta**2) * (precision[i] * recall[i]) / ((beta**2 * precision[i]) + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    return precision, recall, f1_score, f2_score

# Calculate the metrics
precision, recall, f1_score, f2_score = calculate_metrics(optimal_conf_matrix)

# Display the results
for i in range(len(precision)):
    print(f"Class {i}: Precision = {precision[i]:.4f}, Recall = {recall[i]:.4f}, F1 Score = {f1_score[i]:.4f}, F2 Score = {f2_score[i]:.4f}")