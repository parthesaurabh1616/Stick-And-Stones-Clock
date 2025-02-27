#!/usr/bin/env python
# coding: utf-8

# # STEP 1: PARAMETERS AND PATHS

# In[1]:


import os
import re
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

#%% PARAMETERS AND PATHS
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 8
EPOCHS = 50
AUTOTUNE = tf.data.AUTOTUNE


# In[2]:


# Update this path to the location of your dataset
image_folder = "C://Users/saura/OneDrive/Desktop/MothersonProjectTemplate/sundial/dataset"
image_paths = glob.glob(os.path.join(image_folder, "*.*"))


# # STEP 2: LABEL EXTRACTION

# In[3]:


#%% LABEL EXTRACTION
def extract_time_from_filename(filepath):
    """
    Extracts time from the filename in the format HHMM or HH_MM.
    Returns time in decimal hours (e.g., "2030" -> 20.5).
    """
    filename = os.path.basename(filepath)
    match = re.search(r'(\d{2})[\-_]?(\d{2})', filename)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        return hour + minute / 60.0
    else:
        return None

# Prepare valid image paths and corresponding time labels
paths = []
times = []
for path in image_paths:
    t = extract_time_from_filename(path)
    if t is not None:
        paths.append(path)
        times.append(t)
print("Total valid images:", len(paths))


# # STEP 3: CYCLIC ENCODING

# In[4]:


#%% CYCLIC ENCODING OF TIME
def encode_time_cyclic(time_val):
    """
    Encodes time (in hours) into cyclic components using sine and cosine.
    """
    radians = 2 * np.pi * (time_val / 24.0)
    return np.sin(radians), np.cos(radians)

labels_sin = []
labels_cos = []
for t in times:
    sin_val, cos_val = encode_time_cyclic(t)
    labels_sin.append(sin_val)
    labels_cos.append(cos_val)

labels_sin = np.array(labels_sin, dtype=np.float32)
labels_cos = np.array(labels_cos, dtype=np.float32)
# Combine into a single 2D array
labels_combined = np.stack([labels_sin, labels_cos], axis=1)

# Convert paths and labels to tensors
paths_tensor = tf.constant(paths)
labels_tensor = tf.constant(labels_combined, dtype=tf.float32)


# # STEP 4: DATA LOADING AND PREPROCESSING

# In[5]:


#%% DATA LOADING AND PREPROCESSING
def load_and_preprocess(path, label):
    """
    Loads an image from a file path and preprocesses it:
      - Decodes the image.
      - Resizes it to (IMG_HEIGHT, IMG_WIDTH).
      - Normalizes pixel values to [0, 1].
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))
dataset = dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
dataset = dataset.shuffle(len(paths))


# # STEP 5: DATA SPLITTING

# In[6]:


#%% SPLIT DATASET INTO TRAINING AND VALIDATION
val_size = int(0.2 * len(paths))
train_dataset = dataset.skip(val_size)
val_dataset = dataset.take(val_size)


# # STEP 6: DATA AUGMENTATION

# In[7]:


#%% DATA AUGMENTATION (Optional)
def augment(image, label):
    """
    Applies random brightness and contrast adjustments.
    """
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label

train_dataset = train_dataset.map(augment, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# # STEP 7: MODEL BUILDING USING TRANSFER LEARNING

# In[8]:


#%% MODEL BUILDING USING TRANSFER LEARNING (VGG16)
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
# Predict cyclic components (sine and cosine) using tanh activation
predictions = Dense(2, activation='tanh')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])
model.summary()


# # STEP 8: MODEL TRAINING

# In[9]:


#%% TRAINING THE MODEL
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS,
                    callbacks=callbacks)


# # STEP 9: EVALUATION AND PREDICTION

# In[10]:


#%% EVALUATION AND PREDICTION
val_loss, val_mae = model.evaluate(val_dataset)
print("Validation MSE Loss:", val_loss)
print("Validation MAE:", val_mae)

predictions = model.predict(val_dataset)


# # STEP 10: DECODING FUNCTIONS

# In[11]:


#%% DECODING FUNCTIONS
def decode_time(pred):
    """
    Decodes the predicted sine and cosine values into time in hours.
    """
    pred_sin, pred_cos = pred[0], pred[1]
    angle = np.arctan2(pred_sin, pred_cos)
    if angle < 0:
        angle += 2 * np.pi
    time_in_hours = (angle / (2 * np.pi)) * 24
    return time_in_hours


# In[12]:


def decode_label(label):
    """
    Decodes the true label (sine and cosine) back to time in hours.
    """
    sin_val, cos_val = label[0], label[1]
    angle = np.arctan2(sin_val, cos_val)
    if angle < 0:
        angle += 2 * np.pi
    return (angle / (2 * np.pi)) * 24

val_labels = []
for _, lbl in val_dataset.unbatch():
    val_labels.append(decode_label(lbl.numpy()))

print("\nSample Predictions (Decoded):")
for i in range(min(5, len(predictions))):
    pred_time = decode_time(predictions[i])
    print(f"Actual Time: {val_labels[i]:.2f} hrs, Predicted Time: {pred_time:.2f} hrs")


# # STEP 11: PLOTTING TRANING HISTORY

# In[13]:


#%% PLOTTING TRAINING HISTORY
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


# In[14]:


plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




