"""
CIFAR10 Classification with Transfer Learning (ResNet-50 vs VGG-19)

This script demonstrates how to use transfer learning for image classification on the CIFAR10 dataset, comparing two powerful pretrained models: ResNet-50 and VGG-19. The workflow includes:
- Importing libraries
- Loading and preprocessing data
- Visualizing sample images
- Building and training ResNet-50 and VGG-19 models with transfer learning
- Comparing performance and visualizing results
- Confusion matrix analysis
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# Load the CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Resize images to 224x224 for compatibility with pretrained models
IMG_SIZE = 224
train_images_resized = tf.image.resize(train_images, [IMG_SIZE, IMG_SIZE]).numpy()
test_images_resized = tf.image.resize(test_images, [IMG_SIZE, IMG_SIZE]).numpy()

# Normalize pixel values to [0, 1]
train_images_resized = train_images_resized / 255.0
test_images_resized = test_images_resized / 255.0

# Define class names for CIFAR10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Display 25 sample images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    img = train_images_resized[i]
    if img.shape[-1] == 3:
        plt.imshow(img)
    else:
        plt.imshow(img.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(class_names[int(train_labels[i,0])])
plt.tight_layout()
plt.show()

# Prepare labels for training (flatten)
train_labels_flat = train_labels.flatten()
test_labels_flat = test_labels.flatten()

# Load ResNet-50 via keras.applications
base_resnet = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_resnet.trainable = False
resnet_model = keras.Sequential([
    base_resnet,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

# Compile ResNet-50 model
resnet_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

# Train ResNet-50 model
history_resnet = resnet_model.fit(train_images_resized, train_labels_flat, epochs=10,
                                  batch_size=64,
                                  validation_data=(test_images_resized, test_labels_flat),
                                  verbose=2)

# Load VGG-19 via keras.applications
base_vgg = keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_vgg.trainable = False
vgg_model = keras.Sequential([
    base_vgg,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

# Compile VGG model
vgg_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train VGG model
history_vgg = vgg_model.fit(train_images_resized, train_labels_flat, epochs=10,
                            batch_size=64,
                            validation_data=(test_images_resized, test_labels_flat),
                            verbose=2)

# Convert training histories to DataFrames
history_resnet_df = pd.DataFrame(history_resnet.history)
history_vgg_df = pd.DataFrame(history_vgg.history)

# Plot training and validation accuracy for both models
plt.figure(figsize=(10,5))
sns.lineplot(data=history_resnet_df['val_accuracy'], label='ResNet-50 (val)')
sns.lineplot(data=history_vgg_df['val_accuracy'], label='VGG-19 (val)')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.show()

# Evaluate both models on the test set
resnet_test_loss, resnet_test_acc = resnet_model.evaluate(test_images_resized, test_labels_flat, verbose=2)
vgg_test_loss, vgg_test_acc = vgg_model.evaluate(test_images_resized, test_labels_flat, verbose=2)
print(f'ResNet-50 Test Accuracy: {resnet_test_acc:.3f}')
print(f'VGG-19 Test Accuracy: {vgg_test_acc:.3f}')

# Show confusion matrix for both models
resnet_pred_labels = np.argmax(resnet_model.predict(test_images_resized), axis=1)
vgg_pred_labels = np.argmax(vgg_model.predict(test_images_resized), axis=1)
cm_resnet = confusion_matrix(test_labels_flat, resnet_pred_labels)
cm_vgg = confusion_matrix(test_labels_flat, vgg_pred_labels)
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('ResNet-50 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.subplot(1,2,2)
sns.heatmap(cm_vgg, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('VGG-19 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

print("\nSummary:\n")
print("This script compared the performance of ResNet-50 and VGG-19 on the CIFAR10 dataset using transfer learning. You can see the validation accuracy and confusion matrices for both models above. Try adjusting the number of epochs or unfreezing layers for further improvements.")
print("\n---\n")
print("Notebook Workflow Recap:")
print("- Importing Libraries: Essential Python libraries for deep learning, data manipulation, and visualization.")
print("- Loading Data: CIFAR10 dataset with 60,000 color images in 10 classes.")
print("- Preprocessing: Resize images to 224x224 pixels and normalize pixel values.")
print("- Visualization: Display 25 sample images with class names.")
print("- Model Setup: Build ResNet-50 and VGG-19 models with transfer learning and custom classification heads.")
print("- Training: Train both models for 10 epochs.")
print("- Evaluation: Visualize training histories and compare test accuracy.")
print("- Confusion Matrix: Plot confusion matrices for both models.")
print("- Summary: Encourage further experimentation.")
print("\nThis script guides you through applying and comparing state-of-the-art deep learning models on a benchmark dataset, highlighting the power of transfer learning.")

