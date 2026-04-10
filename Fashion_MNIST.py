#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

train_df.head()
train_df.shape, test_df.shape


# In[5]:


X_train = train_df.drop("label", axis=1).values
y_train = train_df["label"].values

X_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values

X_train = X_train / 255.0
X_test = X_test / 255.0

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

plt.figure(figsize=(10, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.tight_layout()
plt.show()


# In[7]:


basic_model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

basic_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_basic = basic_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=32,
    verbose=1
)


# In[9]:


deep_model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

deep_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_deep = deep_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=2,
    batch_size=32,
    verbose=1
)


# In[11]:


def plot_history(history, title):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

plot_history(history_basic, "Basic Model")
plot_history(history_deep, "Deep Model")


# In[13]:


basic_test_loss, basic_test_acc = basic_model.evaluate(X_test, y_test, verbose=0)
deep_test_loss, deep_test_acc = deep_model.evaluate(X_test, y_test, verbose=0)

print("Basic Model Test Accuracy:", basic_test_acc)
print("Deep Model Test Accuracy:", deep_test_acc)


# In[15]:


def plot_confusion_matrix(model, X_test, y_test, title):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.show()

    print(classification_report(y_test, y_pred, target_names=class_names))

plot_confusion_matrix(basic_model, X_test, y_test, "Basic Model Confusion Matrix")
plot_confusion_matrix(deep_model, X_test, y_test, "Deep Model Confusion Matrix")


# In[17]:


y_test_bin = label_binarize(y_test, classes=np.arange(10))

basic_probs = basic_model.predict(X_test)
deep_probs = deep_model.predict(X_test)

basic_auc = roc_auc_score(y_test_bin, basic_probs, multi_class="ovr")
deep_auc = roc_auc_score(y_test_bin, deep_probs, multi_class="ovr")

print("Basic Model Multiclass AUC:", basic_auc)
print("Deep Model Multiclass AUC:", deep_auc)


# In[21]:


experiments = [
    {"lr": 0.001, "batch_size": 64, "epochs": 5, "dropout": 0.2},
    {"lr": 0.0005, "batch_size": 64, "epochs": 5, "dropout": 0.3},
]

results = []

for exp in experiments:
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(exp["dropout"]),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=exp["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=exp["epochs"],
        batch_size=exp["batch_size"],
        verbose=0
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    probs = model.predict(X_test, verbose=0)
    auc = roc_auc_score(y_test_bin, probs, multi_class="ovr")

    results.append({
        "learning_rate": exp["lr"],
        "batch_size": exp["batch_size"],
        "epochs": exp["epochs"],
        "dropout": exp["dropout"],
        "test_accuracy": test_acc,
        "auc": auc
    })

results_df = pd.DataFrame(results)
display(results_df.sort_values(by="test_accuracy", ascending=False))


# In[23]:


comparison_df = pd.DataFrame({
    "Model": ["Basic NN", "Deep NN"],
    "Test Accuracy": [basic_test_acc, deep_test_acc],
    "AUC": [basic_auc, deep_auc]
})

comparison_df

