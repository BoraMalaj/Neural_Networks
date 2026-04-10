# Fashion MNIST - Neural Network Analysis

## Overview
This project involves building, training, and optimizing Deep Learning models to classify images of clothing from the Fashion MNIST dataset. The work is part of a Master's degree course in Machine Learning, focusing on the transition from simple architectures to deep convolutional networks.

## Dataset
- **Source:** Kaggle (Fashion MNIST)
- **Samples:** 60,000 (Training) / 10,000 (Testing)
- **Format:** CSV files (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`)
- **Image Size:** 28x28 pixels (Grayscale)
- **Classes:** 10 categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

## Model Architectures
| Model Type | Key Layers | Purpose |
|------------|------------|---------|
| **Basic (MLP)** | Flatten, Dense (128 units), Softmax | Established baseline performance |
| **Deep (CNN)** | Conv2D, MaxPooling, Dropout, Dense | Spatial feature extraction & high accuracy |

## Tasks & Steps Covered
1. **Data Preprocessing:** - Loading local Kaggle CSV data via Pandas.
   - Normalization (scaling pixel values from [0, 255] to [0, 1]).
   - Reshaping data into (28, 28, 1) tensors for CNN compatibility.
2. **Model Implementation:** - Building and compiling models using the Keras Sequential API.
3. **Evaluation & Visualization:** - Comparison of Training vs. Validation accuracy/loss.
   - **Confusion Matrix:** Identifying which classes (e.g., Shirt vs. T-shirt) the model confuses.
   - **AUC (Area Under Curve):** Multi-class evaluation of model discriminative power.
   - **Error Visualization:** Plotting misclassified images for qualitative analysis.
4. **Hyperparameter Tuning:** - Testing different **Learning Rates** (Adam Optimizer).
   - Comparing **Batch Sizes** (32 vs 64).
   - Implementing **Regularization** (Dropout & Batch Normalization) to reduce overfitting.

## Libraries Used
- **Modeling:** `tensorflow`, `keras`
- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Metrics:** `scikit-learn`

## Author
Bora Malaj
