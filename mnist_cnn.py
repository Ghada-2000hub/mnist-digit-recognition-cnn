"""Handwritten digit recognition with a CNN on MNIST (with confusion matrix)."""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ------------------------------------------------------------------
# 1) Load and prepare the data
# ------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add the channel dimension: (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# ------------------------------------------------------------------
# 2) Build the CNN
# ------------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ------------------------------------------------------------------
# 3) Train
# ------------------------------------------------------------------
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
)

# Training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

# ------------------------------------------------------------------
# 4) Evaluate on the test set
# ------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc * 100:.2f}%")

# ------------------------------------------------------------------
# 5) Confusion matrix
# ------------------------------------------------------------------
y_pred = np.argmax(model.predict(x_test), axis=1)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(
    ax=ax, cmap="Blues", colorbar=False
)
plt.title("Confusion matrix (MNIST test set)")
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

# ------------------------------------------------------------------
# 6) Which digits does the model confuse the most?
# ------------------------------------------------------------------
cm_errors = cm.copy()
np.fill_diagonal(cm_errors, 0)  # ignore correct predictions

pairs = []
for true_digit in range(10):
    for pred_digit in range(10):
        if cm_errors[true_digit, pred_digit] > 0:
            pairs.append((cm_errors[true_digit, pred_digit], true_digit, pred_digit))
pairs.sort(reverse=True)

print("Most frequent mistakes:")
for count, true_digit, pred_digit in pairs[:5]:
    print(f"  True {true_digit} predicted as {pred_digit}: {count} times")

# Show some misclassified images
wrong = np.where(y_pred != y_test)[0]
plt.figure(figsize=(10, 4))
for i, idx in enumerate(wrong[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx].squeeze(), cmap="gray")
    plt.title(f"True: {y_test[idx]} | Pred: {y_pred[idx]}", fontsize=9)
    plt.axis("off")
plt.suptitle("Misclassified examples")
plt.tight_layout()
plt.savefig("misclassified_examples.png", dpi=150)
plt.show()

# ------------------------------------------------------------------
# 7) Save the model
# ------------------------------------------------------------------
model.save("mnist_cnn_model.keras")
print("\nModel saved as mnist_cnn_model.keras")
