from pathlib import Path
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# CHARGEMENT DU DATASET

csv_path = Path("diabetes.csv")

if not csv_path.exists():
    raise FileNotFoundError(
        "Le fichier diabetes.csv est introuvable. "
        "Place-le dans le même dossier que ce script."
    )

df = pd.read_csv(csv_path)

print("=== PIMA DIABETES — PHASE 5 ===")
print("Shape initiale :", df.shape)


# TRAITEMENT DES ZÉROS SUSPECTS : Dans Pima, certaines valeurs manquantes sont codées avec 0 alors que 0 n'est pas physiologiquement logique. On remplace ces zéros par la médiane des valeurs non nulles.

zero_suspect_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

print("\n=== REMPLACEMENT DES ZÉROS SUSPECTS ===")

for column in zero_suspect_columns:
    zero_count = (df[column] == 0).sum()

    median_without_zero = df.loc[
        df[column] != 0,
        column
    ].median()

    print(
        f"{column:20s} | "
        f"zéros : {zero_count:3d} | "
        f"médiane : {median_without_zero:.2f}"
    )

    df[column] = df[column].replace(
        0,
        median_without_zero
    )


#  SÉPARATION X / y

X = df.drop(
    columns=["Outcome"]
).copy()

y = df["Outcome"].astype(int).copy()


print("\n=== DISTRIBUTION DES CLASSES ===")
print(y.value_counts().sort_index())


# SPLIT TRAIN / VALIDATION / TEST

X_train_val, X_test, y_train_val, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

X_train, X_val, y_train, y_val = (
    train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=42,
        stratify=y_train_val
    )
)


print("\n=== SHAPES ===")
print("Train      :", X_train.shape)
print("Validation :", X_val.shape)
print("Test       :", X_test.shape)


# NORMALISATION

scaler = StandardScaler()

X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)


# CONSTRUCTION DU MODÈLE

def build_pima_regularized(
    l2_lambda=0.01,
    use_dropout=False
):

    model = keras.Sequential()

    model.add(
        keras.Input(shape=(8,))
    )

    model.add(
        keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(
                l2_lambda
            )
        )
    )

    if use_dropout:
        model.add(
            keras.layers.Dropout(0.3)
        )

    model.add(
        keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(
                l2_lambda
            )
        )
    )

    if use_dropout:
        model.add(
            keras.layers.Dropout(0.3)
        )

    model.add(
        keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


#  FONCTION D'ENTRAÎNEMENT

def train_configuration(
    name,
    l2_lambda,
    use_dropout
):


    keras.backend.clear_session()
    keras.utils.set_random_seed(42)

    model = build_pima_regularized(
        l2_lambda=l2_lambda,
        use_dropout=use_dropout
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    print(
        f"\n=== ENTRAÎNEMENT : {name} ==="
    )

    history = model.fit(
        X_train_norm,
        y_train,
        epochs=300,
        batch_size=32,
        validation_data=(
            X_val_norm,
            y_val
        ),
        callbacks=[
            early_stopping
        ],
        verbose=0
    )

    epochs_reels = len(
        history.history["val_loss"]
    )

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    best_val_loss = min(
        history.history["val_loss"]
    )

    test_loss, test_accuracy = model.evaluate(
        X_test_norm,
        y_test,
        verbose=0
    )

    val_probabilities = model.predict(
        X_val_norm,
        verbose=0
    ).flatten()

    print(
        f"Epochs réels       : {epochs_reels}"
    )

    print(
        f"Meilleure val_loss : {best_val_loss:.4f}"
    )

    print(
        f"Max val_accuracy   : {best_val_accuracy:.4f}"
    )

    print(
        f"Test accuracy      : {test_accuracy:.4f}"
    )

    print(
        f"Moyenne prédictions validation : "
        f"{val_probabilities.mean():.4f}"
    )

    return {
        "name": name,
        "model": model,
        "history": history,
        "epochs": epochs_reels,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss
    }


# CONFIGURATION 1 : BASELINE

baseline_result = train_configuration(
    name="Baseline",
    l2_lambda=0.0,
    use_dropout=False
)


#  CONFIGURATION 2 : L2 SEULE

l2_result = train_configuration(
    name="L2 seule",
    l2_lambda=0.01,
    use_dropout=False
)


#  CONFIGURATION 3 : L2 + DROPOUT

l2_dropout_result = train_configuration(
    name="L2 + Dropout",
    l2_lambda=0.01,
    use_dropout=True
)


#  TABLEAU COMPARATIF

results = [
    baseline_result,
    l2_result,
    l2_dropout_result
]

print(
    "\n=== TABLEAU COMPARATIF ==="
)

print(
    f"{'Configuration':18s} | "
    f"{'Epochs':6s} | "
    f"{'Max val_acc':11s} | "
    f"{'Test acc':8s}"
)

print("-" * 58)

for result in results:
    print(
        f"{result['name']:18s} | "
        f"{result['epochs']:6d} | "
        f"{result['best_val_accuracy']:.4f}      | "
        f"{result['test_accuracy']:.4f}"
    )


# COURBES VAL_LOSS CÔTE À CÔTE

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

for axis, result in zip(
    axes,
    results
):
    val_losses = (
        result["history"].history["val_loss"]
    )

    axis.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="Validation loss"
    )

    axis.axvline(
        x=result["epochs"],
        linestyle="--",
        label="Arrêt"
    )

    axis.set_title(
        result["name"]
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Val loss")
    axis.grid()
    axis.legend()


plt.tight_layout()

plt.savefig(
    "phase5_pima_3configs.png",
    dpi=120,
    bbox_inches="tight"
)

plt.close()


print(
    "\nCourbes enregistrées dans : "
    "phase5_pima_3configs.png"
)