#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Replace these with the actual paths where you saved your Kaggle files
train_path = 'fashion-mnist_train.csv'
test_path = 'fashion-mnist_test.csv'

# Load the datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Separate labels (y) and pixel values (X)
# The first column 'label' is our target; the rest are the 784 pixels
y_train = train_df['label'].values
x_train = train_df.drop('label', axis=1).values

y_test = test_df['label'].values
x_test = test_df.drop('label', axis=1).values

# Normalization: Scale pixels to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape for the Deep CNN model (28x28 pixels with 1 color channel)
x_train_reshaped = x_train.reshape(-1, 28, 28, 1)
x_test_reshaped = x_test.reshape(-1, 28, 28, 1)

print(f"Data loaded! Training shape: {x_train_reshaped.shape}")


# In[3]:


import tensorflow as tf
from tensorflow.keras import layers, models

def build_basic_model():
    model = models.Sequential([
        layers.Input(shape=(784,)), # Input is the flat vector from the CSV
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

basic_model = build_basic_model()
# Note: We use x_train (flat) here, not the reshaped version
basic_model.fit(x_train, y_train, epochs=2, batch_size=32, validation_split=0.2)


# In[7]:


def build_deep_model():
# The "Modern" Keras way to avoid that warning:
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)), # Explicit Input layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

deep_model = build_deep_model()
# Note: We use x_train_reshaped here
deep_model.fit(x_train_reshaped, y_train, epochs=3, batch_size=32, validation_split=0.2)


# In[9]:


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Get predictions
y_pred_probs = deep_model.predict(x_test_reshaped)
y_pred = np.argmax(y_pred_probs, axis=1)

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 3. AUC (Area Under Curve) - Multi-class
# We use 'ovr' (One-vs-Rest) for multi-class AUC
auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')
print(f"Total AUC Score: {auc:.4f}")


# In[11]:


# Find indices where prediction != actual
errors = np.where(y_pred != y_test)[0]

plt.figure(figsize=(12, 5))
for i, idx in enumerate(errors[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test_reshaped[idx].reshape(28,28), cmap='gray')
    plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}")
    plt.axis('off')
plt.show()


# In[13]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, BatchNormalization

def build_tuned_model(learning_rate=0.001, dropout_rate=0.2):
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(), # Added for stability
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Dropout(dropout_rate), # Regularization
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Example: Try a smaller learning rate and larger batch size
tuned_model = build_tuned_model(learning_rate=0.0005, dropout_rate=0.3)
history_tuned = tuned_model.fit(
    x_train_reshaped, y_train, 
    epochs=10, 
    batch_size=64, # Increased batch size
    validation_split=0.2
)

