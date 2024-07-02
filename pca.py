import numpy as np
import gzip
import matplotlib.pyplot as plt

### Flattening ###
# Function to read pixel data from the dataset
def read_pixels(data_path):
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), 'B', offset=16).astype('float32')
    normalized_pixels = pixel_data / 255
    flattened_pixels = normalized_pixels.reshape(-1, 28 * 28)
    return flattened_pixels

# Function to read label data from the dataset
def read_labels(data_path):
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), 'B', offset=8)
    return label_data

# Read pixel and label data
images = read_pixels("data/train-images-idx3-ubyte.gz")
labels = read_labels("data/train-labels-idx1-ubyte.gz")

# Display the shape of the flattened pixel data
print("Shape of the flattened pixel data:", images.shape)

### Question 1.1 ###
# Function to perform PCA
def apply_pca(data, number_of_components):
    # Calculate the covariance matrix
    covariance_matrix = np.cov(data, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and select top k eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_indices = sorted_indices[:number_of_components]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate PVE
    total_variance = np.sum(eigenvalues)
    pve = np.cumsum(eigenvalues[top_indices]) / total_variance

    return sorted_eigenvectors, pve

# Apply PCA for the first 10 principal components
number_of_components = 10
principal_components, pve = apply_pca(images, number_of_components)

# Display the proportion of variance explained for the first 10 principal components
print("Proportion of Variance Explained (PVE) for the first 10 principal components:")
for i in range(number_of_components):
    print(f"PC{i + 1}: {np.real(pve[i]):.8f}")

### Question 1.2 ###
# Function to perform PCA and determine the number of components needed to explain a threshold PVE
def find_components_for_threshold(data, threshold):
    # Calculate the covariance matrix
    covariance_matrix = np.cov(data, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, _ = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]

    # Calculate PVE
    total_variance = np.sum(eigenvalues)
    cumulative_pve = np.cumsum(eigenvalues[sorted_indices]) / total_variance

    # Find the number of components needed to explain the threshold
    number_of_components = np.argmax(cumulative_pve >= threshold) + 1

    return number_of_components, cumulative_pve

# Specify the threshold for cumulative PVE
threshold = 0.7

# Apply PCA to find the number of components needed to explain 70% of the data
number_of_components_needed, cumulative_pve = find_components_for_threshold(images, threshold)

# Display the results
print(f"Number of components needed to explain {threshold * 100}% of the data: {number_of_components_needed}")
print("Cumulative PVE for each component:")
for i in range(number_of_components_needed):
    print(f"PC{i + 1}: {np.real(cumulative_pve[i]):.8f}")

### Question 1.3 ###
# Convert principal components to real-valued
real_principal_components = np.real(principal_components)

# Reshape and scale the principal components
scaled_principal_components = (real_principal_components - np.min(real_principal_components)) / (np.max(real_principal_components) - np.min(real_principal_components))

# Set the number of components to display
number_of_components = 10

# Set the number of rows and columns for the plot grid
number_of_rows = 2
number_of_cols = 5

# Display the obtained grayscale principal component images
plt.figure(figsize=(15, 6))
for i in range(number_of_components):
    plt.subplot(number_of_rows, number_of_cols, i + 1)
    pc_image = scaled_principal_components[:, i].reshape(28, 28)
    plt.imshow(pc_image, cmap='gray')
    plt.title(f'PC{i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

### Question 1.4 ###
# Function onto project data to principal components
def project_onto_principal_component(data, principal_components, number_of_components):
    return np.dot(data, principal_components[:, :number_of_components])

# Project onto the first 2 principal components
number_of_data_points = 100
projected_data = project_onto_principal_component(images[:number_of_data_points, :], principal_components, number_of_components=2)

# Plot the projected data points
plt.figure(figsize=(10, 6))
for digit_label in range(10):
    indices = np.where(labels[:number_of_data_points] == digit_label)
    plt.scatter(projected_data[indices, 0], projected_data[indices, 1], label=str(digit_label))

plt.title('Projection of the First 100 Images onto the First 2 Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Digit Label')
plt.grid(True)
plt.show()

### Question 1.5 ###
# Normalize the eigenvectors to have unit norm
normalized_eigenvectors = principal_components / np.linalg.norm(principal_components, axis=0)

# Function to reconstruct an image using the first k principal components
def reconstruct_image(image, mean_image, principal_components, k):
    # Project the image onto the first k principal components
    weights = np.dot(image - mean_image, principal_components[:, :k])

    # Reconstruct the image using the weighted sum
    reconstructed_image = mean_image + np.dot(weights, principal_components[:, :k].T)

    return np.real(reconstructed_image)

# Choose the first image in the dataset
image_1 = images[0, :]
mean_image = np.mean(images, axis=0)

# Specify the values of k for reconstruction
k_values = [1, 50, 100, 250, 500, 784]

# Plot the original and reconstructed images for different values of k
plt.figure(figsize=(15, 7))
for i, k in enumerate(k_values, 1):
    reconstructed_image = reconstruct_image(image_1, mean_image, normalized_eigenvectors, k)

    # Plot the original image
    plt.subplot(2, len(k_values), i)
    plt.imshow(image_1.reshape(28, 28), cmap='gray')
    plt.title(f'Original Image')
    plt.axis('off')

    # Plot the reconstructed image
    plt.subplot(2, len(k_values), i + len(k_values))
    plt.imshow(reconstructed_image.reshape(28, 28), cmap='gray')
    plt.title(f'Reconstructed Image (k={k})')
    plt.axis('off')

plt.tight_layout()
plt.show()