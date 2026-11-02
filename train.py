"""
Train a 6-class CNN for visual air-pollution image classification.

Dataset source and folder setup are explained in dataset/DATASET_INFO.txt.
Install: pip install -r requirements.txt
Run:     python train.py

Expected class folders:
dataset/Combined_Dataset/IND_and_NEP/
    Good/
    Moderate/
    Unhealthy for Sensitive Groups/
    Unhealthy/
    Very Unhealthy/
    Hazardous/
"""
import os, re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = "dataset/Combined_Dataset/IND_and_NEP"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42
OUTPUT_DIR = "outputs"
CLASS_NAMES = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
               "Unhealthy", "Very Unhealthy", "Hazardous"]
ALIASES = {
    "good":"Good", "moderate":"Moderate",
    "unhealthyforsensitivegroups":"Unhealthy for Sensitive Groups",
    "unhealthyforsensitivegroup":"Unhealthy for Sensitive Groups",
    "unhealthy":"Unhealthy", "veryunhealthy":"Very Unhealthy",
    "hazardous":"Hazardous", "severe":"Hazardous"
}
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def normalize_folder(text):
    return re.sub(r"[^a-z]", "", text.lower())

def collect_images():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}. Read dataset/DATASET_INFO.txt."
        )
    rows = []
    for root, _, files in os.walk(DATASET_DIR):
        for name in files:
            if name.lower().endswith(EXTS):
                parts = os.path.relpath(root, DATASET_DIR).split(os.sep)
                label = None
                for part in reversed(parts):
                    key = normalize_folder(part)
                    if key in ALIASES:
                        label = ALIASES[key]
                        break
                if label:
                    rows.append({"filepath": os.path.join(root, name), "label": label})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No labelled images found. Check the folder structure.")
    missing = [c for c in CLASS_NAMES if c not in set(df.label)]
    if missing:
        raise ValueError("Missing class folders: " + ", ".join(missing))
    return df

def create_generators(train_df, val_df, test_df):
    train_aug = ImageDataGenerator(rescale=1/255.0, rotation_range=8,
                                   zoom_range=0.10, horizontal_flip=True)
    normal = ImageDataGenerator(rescale=1/255.0)
    common = dict(x_col="filepath", y_col="label", classes=CLASS_NAMES,
                  target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                  class_mode="categorical")
    train = train_aug.flow_from_dataframe(train_df, shuffle=True, seed=SEED, **common)
    val = normal.flow_from_dataframe(val_df, shuffle=False, **common)
    test = normal.flow_from_dataframe(test_df, shuffle=False, **common)
    return train, val, test

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128,128,3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.40),
        tf.keras.layers.Dense(6, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    df = collect_images()
    print("Images per class:\\n", df.label.value_counts().reindex(CLASS_NAMES, fill_value=0))
    train_df, remaining = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df.label)
    val_df, test_df = train_test_split(
        remaining, test_size=0.50, random_state=SEED, stratify=remaining.label)
    pd.DataFrame({"split":["train","validation","test"],
                  "images":[len(train_df),len(val_df),len(test_df)]}
                 ).to_csv(f"{OUTPUT_DIR}/dataset_split_summary.csv", index=False)

    train, val, test = create_generators(train_df, val_df, test_df)
    model = build_model()
    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(train, validation_data=val, epochs=EPOCHS,
                        callbacks=[early_stop])
    loss, accuracy = model.evaluate(test, verbose=1)
    model.save(f"{OUTPUT_DIR}/air_pollution_cnn.keras")

    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].plot(history.history["accuracy"], label="Train")
    ax[0].plot(history.history["val_accuracy"], label="Validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="Train")
    ax[1].plot(history.history["val_loss"], label="Validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/training_curves.png", dpi=150); plt.close()

    probs = model.predict(test, verbose=1)
    pred = np.argmax(probs, axis=1)
    actual = test.classes
    report = classification_report(actual, pred, labels=list(range(6)),
                                   target_names=CLASS_NAMES, zero_division=0)
    with open(f"{OUTPUT_DIR}/classification_report.txt","w",encoding="utf-8") as f:
        f.write(f"Test loss: {loss:.4f}\\nTest accuracy: {accuracy:.4f}\\n\\n{report}")
    cm = confusion_matrix(actual, pred, labels=list(range(6)))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10,8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150); plt.close()
    with open(f"{OUTPUT_DIR}/experiment_summary.txt","w") as f:
        f.write(f"Images: {len(df)}\\nTrain: {len(train_df)}\\nValidation: {len(val_df)}\\n"
                f"Test: {len(test_df)}\\nTest loss: {loss:.4f}\\nTest accuracy: {accuracy:.4f}\\n")
    print(f"Finished. Test accuracy = {accuracy:.4f}")
    print("Model and evaluation files saved in outputs/")

if __name__ == "__main__":
    main()
