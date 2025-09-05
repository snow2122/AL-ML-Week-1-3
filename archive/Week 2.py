#quiet logs
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

import tensorflow as tf

#data loading
DATA_DIR  = r"C:\Users\Meerab\Downloads\internsip\subset_500"
IMG_SIZE  = (128, 128)
BATCH     = 32
SEED      = 42

train_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,     # also resizes to 128x128
    batch_size=BATCH,
    label_mode="binary"
)

val_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="binary"
)

AUTOTUNE = tf.data.AUTOTUNE

#normalize to [0,1]
rescale = tf.keras.layers.Rescaling(1./255)

train_ds_cnn = train_raw.map(lambda x, y: (rescale(x), y)).cache().prefetch(AUTOTUNE)
val_ds_cnn   = val_raw.map(  lambda x, y: (rescale(x), y)).cache().prefetch(AUTOTUNE)

#Scratch CNN (3 conv layers)
cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.summary()

print("Training scratch CNN (10 epochs)…")
hist_cnn = cnn.fit(train_ds_cnn, validation_data=val_ds_cnn, epochs=10)
val_loss_cnn, val_acc_cnn = cnn.evaluate(val_ds_cnn)
print(f"[Scratch CNN] Val Acc: {val_acc_cnn:.3f} | Val Loss: {val_loss_cnn:.3f}")


#use VGG16 preprocessing, NOT Rescaling
# VGG16 expects pixels in a special range/format; use its preprocess_input
vgg_prep = tf.keras.applications.vgg16.preprocess_input

train_ds_vgg = train_raw.map(lambda x, y: (vgg_prep(x), y)).cache().prefetch(AUTOTUNE)
val_ds_vgg   = val_raw.map(  lambda x, y: (vgg_prep(x), y)).cache().prefetch(AUTOTUNE)


#VGG16 TRANSFER LEARNING
base = tf.keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)
base.trainable = False  # freeze feature extractor

vgg16_model = tf.keras.Sequential([
    base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

vgg16_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training VGG16 transfer model (5 epochs)…")
hist_vgg = vgg16_model.fit(train_ds_vgg, validation_data=val_ds_vgg, epochs=5)
val_loss_vgg, val_acc_vgg = vgg16_model.evaluate(val_ds_vgg)
print(f"[VGG16 TL]   Val Acc: {val_acc_vgg:.3f} | Val Loss: {val_loss_vgg:.3f}")
