import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Path to dataset (place ISIC Skin Cancer Dataset in project folder)
data_path = r"C:\Users\Meerab\Downloads\internsip\ISIC Skin Cancer Dataset"
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

# Image preprocessing
img_size = 128
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

# Load pre-trained ResNet50
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    steps_per_epoch=min(len(train_gen), 30),  # limit steps for ~500-1000 images
    validation_steps=min(len(val_gen), 10)
)

# Evaluate
loss, accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {accuracy:.4f}")

import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# dataset path
data_path = r"C:\Users\Meerab\Downloads\internsip\Kaggle Chest X-Ray Images Dataset"

# parameters
img_size = (128, 128)
batch = 32
seed = 42

# load train/test datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_path, "train"),
    image_size=img_size,
    batch_size=batch,
    label_mode="binary",
    seed=seed
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(data_path, "test"),
    image_size=img_size,
    batch_size=batch,
    label_mode="binary",
    seed=seed
)

# take subset first 500 images from each set
train_ds = train_ds.unbatch().take(500).batch(batch)
test_ds = test_ds.unbatch().take(200).batch(batch)

# normalize [0,1]
rescale = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (rescale(x), y))
test_ds = test_ds.map(lambda x, y: (rescale(x), y))

# simple CNN (2 conv layers)
cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.summary()

# train model
print("Training CNN for Pneumonia Detection (5 epochs)…")
history = cnn.fit(train_ds, validation_data=test_ds, epochs=5)

# evaluate accuracy
loss, acc = cnn.evaluate(test_ds)
print(f"Pneumonia CNN Test Accuracy: {acc:.3f}")

# ROC curve
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = cnn.predict(test_ds).ravel()

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'CNN (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Pneumonia Detection')
plt.legend(loc="lower right")
plt.show()
