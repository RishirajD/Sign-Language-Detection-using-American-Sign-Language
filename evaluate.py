import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, silhouette_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

# Model path
MODEL_PATH = './models/asl_inception_v3_trainedpro.h5'

# Dataset paths
TRAIN_PATH = 'D:/objdetection_inceptionmodel/data/train'
TEST_PATH = 'D:/objdetection_inceptionmodel/data/test'

# Hyperparameters
IMAGE_SIZE = (299, 299)  # InceptionV3 input size
BATCH_SIZE = 32

# Load the pretrained model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Predictions and ground truth labels
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Calculate F1 Score
f1 = f1_score(true_classes, predicted_classes, average='weighted')
print(f"F1 Score: {f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Silhouette Score
# Silhouette score is typically used for clustering, but we can repurpose it to evaluate class separation
flat_features = np.argmax(predictions, axis=1).reshape(-1, 1)  # Flatten predictions
silhouette = silhouette_score(flat_features, true_classes)
print(f"Silhouette Score: {silhouette:.4f}")

# Training History Visualization
history = model.history.history  # If training history is saved, load it
if history:
    plt.figure(figsize=(12, 6))

    # Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

else:
    print("Training history is unavailable. Skipping accuracy/loss visualization.")

# Testing Accuracy Visualization
batch_accuracies = []
for i in range(len(test_generator)):
    batch_x, batch_y = test_generator[i]
    batch_preds = model.predict(batch_x)
    batch_predicted_classes = np.argmax(batch_preds, axis=1)
    batch_true_classes = np.argmax(batch_y, axis=1)
    batch_accuracy = accuracy_score(batch_true_classes, batch_predicted_classes)
    batch_accuracies.append(batch_accuracy)

plt.figure(figsize=(10, 5))
plt.plot(range(len(batch_accuracies)), batch_accuracies, marker='o')
plt.title("Batch-wise Testing Accuracy")
plt.xlabel("Batch Index")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

print("Model evaluation completed!")
