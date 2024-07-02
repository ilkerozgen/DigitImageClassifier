# Digit Image Classifier

## Description

This project is divided into two main parts: PCA Analysis and Logistic Regression Classification. The objective is to analyze the MNIST dataset and classify digit images using Multinomial Logistic Regression.

### Part 1: PCA Analysis

1. **Principal Component Analysis (PCA)**
   - **Dataset**: MNIST dataset, containing 70,000 grayscale digit images (60,000 training and 10,000 test images).
   - **Image Resolution**: 28x28 pixels.
   
   **Tasks**:
   - Apply PCA and report the proportion of variance explained (PVE) for the first 10 principal components.
   - Report the number of principal components required to explain 70% of the data.
   - Visualize the first 10 principal components by reshaping them into 28x28 images and applying min-max scaling.
   - Project the first 100 images onto the first 2 principal components and plot the data points colored by their labels.
   - Reconstruct an original digit image using the first k principal components (k ∈ {1, 50, 100, 250, 500, 784}).

### Part 2: Logistic Regression

2. **Multinomial Logistic Regression Classifier**
   - **Dataset**: MNIST dataset, divided into 50,000 training, 10,000 validation, and 10,000 test images.
   
   **Tasks**:
   - Train a default logistic regression model and display the test accuracy and confusion matrix.
   - Experiment with different hyperparameters (batch size, weight initialization, learning rate, regularization coefficient) and compare performances.
   - Select the best hyperparameters, create the optimal model, and display the test accuracy and confusion matrix.
   - Visualize the finalized weight vectors as images.
   - Calculate precision, recall, F1 score, and F2 score for each class using the best model and comment on the results.

## Dataset

- **Source**: MNIST dataset
- **Total Images**: 70,000 (60,000 training and 10,000 test images)
- **Image Resolution**: 28x28 pixels
- **Labels**: Handwritten digits (0-9)

> **Note**: Due to the size of the dataset, it cannot be shared on GitHub. Please contact the project owner for access to the dataset.

## Tech Stack

- **Programming Language**: Python
- **Libraries**: No additional machine learning libraries are used; all implementations are self-contained.

## Installation

To run the project, follow these steps:

1. On the command line, navigate to the directory where the `q1.py` and `q2.py` files are located.
2. To run each script, type:
   ```bash
   python pca.py
   ```
   or
   ```bash
   python classifier.py
   ```
3. No additional parameters are needed. The results will be printed on the command line, and plots will appear as new windows.

## Results and Analysis

For detailed results and a comprehensive analysis of our findings, please refer to the [report.pdf](report.pdf).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contribution

Contributions are welcome! Please fork this repository and submit pull requests for any improvements or bug fixes.
