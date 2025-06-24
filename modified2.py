import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score

# Constants
DATA_DIR = r"images/train"
LABELS_FILE =  r"train.json"
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS =200

# Load labels from JSON file
def load_labels(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# Preprocess image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img_array

# Load dataset
def load_dataset(data_dir, labels):
    images = []
    quaternions = []
    translations = []

    for entry in labels:
        img_path = os.path.join(data_dir, entry["filename"])
        if os.path.exists(img_path):
            images.append(preprocess_image(img_path))
            quaternions.append(entry["q_vbs2tango"])
            translations.append(entry["r_Vo2To_vbs_true"])
    
    return np.array(images), np.array(quaternions), np.array(translations)

# Load labels and dataset
labels = load_labels(LABELS_FILE)
images, quaternions, translations = load_dataset(DATA_DIR, labels)

# Split data
X_train, X_val, y_train_quat, y_val_quat, y_train_trans, y_val_trans = train_test_split(
    images, quaternions, translations, test_size=0.2, random_state=42
)

# Define the model
from tensorflow.keras import layers, Model

def create_pose_model(input_shape):
    # Base model
    base_model = tf.keras.applications.ResNet50(
        include_top=False, input_shape=input_shape, weights="imagenet"
    )
    base_model.trainable = False  # Freeze base model

    # Feature extraction
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Quaternion prediction head
    quaternion_head = layers.Dense(256, activation="relu")(x)
    quaternion_head = layers.Dense(4, activation="linear", name="quaternion")(quaternion_head)

    # Translation prediction head
    translation_head = layers.Dense(256, activation="relu")(x)
    translation_head = layers.Dense(3, activation="linear", name="translation")(translation_head)

    # Final model
    model = Model(inputs=base_model.input, outputs=[quaternion_head, translation_head])
    return model

# Create the model
pose_model = create_pose_model((IMG_HEIGHT, IMG_WIDTH, 3))
pose_model.summary()

# Define custom loss for quaternions
def quaternion_loss(y_true, y_pred):
    y_pred = tf.nn.l2_normalize(y_pred, axis=-1)  # Normalize quaternion
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Compile the model
pose_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={"quaternion": quaternion_loss, "translation": "mse"},
    metrics={"quaternion": "mse", "translation": "mae"}
)

# Train the model
history = pose_model.fit(
    X_train,
    {"quaternion": y_train_quat, "translation": y_train_trans},
    validation_data=(X_val, {"quaternion": y_val_quat, "translation": y_val_trans}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Evaluate the model
results = pose_model.evaluate(X_val, {"quaternion": y_val_quat, "translation": y_val_trans})
print("Validation Results:", results)

# Function to compute Geodesic Distance for quaternions
def geodesic_distance(q_true, q_pred):
    dot_product = np.sum(q_true * q_pred, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid range for arccos
    return 2 * np.arccos(np.abs(dot_product))

# Predict on validation set
y_pred_quat, y_pred_trans = pose_model.predict(X_val)

# Convert to numpy arrays
y_val_quat = np.array(y_val_quat)
y_val_trans = np.array(y_val_trans)

# Compute metrics for quaternion predictions
mae_quat = mean_absolute_error(y_val_quat, y_pred_quat)
mse_quat = mean_squared_error(y_val_quat, y_pred_quat)
rmse_quat = np.sqrt(mse_quat)
r2_quat = r2_score(y_val_quat, y_pred_quat)
mape_quat = mean_absolute_percentage_error(y_val_quat, y_pred_quat)
evs_quat = explained_variance_score(y_val_quat, y_pred_quat)
geo_dist_quat = np.mean(geodesic_distance(y_val_quat, y_pred_quat))

# Compute metrics for translation predictions
mae_trans = mean_absolute_error(y_val_trans, y_pred_trans)
mse_trans = mean_squared_error(y_val_trans, y_pred_trans)
rmse_trans = np.sqrt(mse_trans)
r2_trans = r2_score(y_val_trans, y_pred_trans)
mape_trans = mean_absolute_percentage_error(y_val_trans, y_pred_trans)
evs_trans = explained_variance_score(y_val_trans, y_pred_trans)

# Print results
print("Quaternion (Orientation) Metrics:")
print(f"MAE: {mae_quat:.6f}, MSE: {mse_quat:.6f}, RMSE: {rmse_quat:.6f}, R²: {r2_quat:.6f}, MAPE: {mape_quat:.6f}, EVS: {evs_quat:.6f}, Geodesic Distance: {geo_dist_quat:.6f}")

print("\nTranslation (Position) Metrics:")
print(f"MAE: {mae_trans:.6f}, MSE: {mse_trans:.6f}, RMSE: {rmse_trans:.6f}, R²: {r2_trans:.6f}, MAPE: {mape_trans:.6f}, EVS: {evs_trans:.6f}")

# Save the model
pose_model.save("satellite_pose_model.h5")

# Plot training history
plt.figure(figsize=(12, 6))

# Check if the keys exist in history before plotting
if "quaternion_loss" in history.history:
    plt.plot(history.history["quaternion_loss"], label="Quaternion Loss (Train)")
if "val_quaternion_loss" in history.history:
    plt.plot(history.history["val_quaternion_loss"], label="Quaternion Loss (Val)")
if "translation_loss" in history.history:
    plt.plot(history.history["translation_loss"], label="Translation Loss (Train)")
if "val_translation_loss" in history.history:
    plt.plot(history.history["val_translation_loss"], label="Translation Loss (Val)")
if "loss" in history.history:
    plt.plot(history.history["loss"], label="Total Loss (Train)")
if "val_loss" in history.history:
    plt.plot(history.history["val_loss"], label="Total Loss (Val)")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()
