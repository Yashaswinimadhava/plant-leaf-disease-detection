
from pathlib import Path
import json

import tensorflow as tf

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


def main():
    # Check that the dataset exists
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR.resolve()}"
        )

    class_folders = sorted(
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if len(class_folders) < 2:
        raise ValueError(
            "The dataset must contain at least two class folders."
        )

    # Create output folder before saving any files
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PlantVillage dataset...")

    # Load training images
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    # Load validation images using the same split
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_ds.class_names

    # Save labels in exactly the order used by the model
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=4)

    print(f"\nNumber of classes: {len(class_names)}")
    print("Classes:")
    for index, name in enumerate(class_names):
        print(f"{index}: {name}")

    # Improve input pipeline performance
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    # -----------------------------
    # Build CNN model
    # -----------------------------
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),

        # Normalize pixel values from 0-255 to 0-1
        tf.keras.layers.Rescaling(1.0 / 255),

        # Data augmentation for training
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),

        tf.keras.layers.Conv2D(
            32, kernel_size=3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64, kernel_size=3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128, kernel_size=3, activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # One output for each disease class
        tf.keras.layers.Dense(
            len(class_names), activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # -----------------------------
    # Training callbacks
    # -----------------------------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # -----------------------------
    # Train model
    # -----------------------------
    print("\nStarting model training...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save the final model with the restored best weights
    model.save(MODEL_PATH)

    print("\nTraining completed!")
    print(f"Model saved to: {MODEL_PATH.resolve()}")
    print(f"Class labels saved to: {CLASS_NAMES_PATH.resolve()}")

    final_train_accuracy = history.history["accuracy"][-1]
    final_val_accuracy = history.history["val_accuracy"][-1]

    print(f"Last training accuracy: {final_train_accuracy:.4f}")
    print(f"Last validation accuracy: {final_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
