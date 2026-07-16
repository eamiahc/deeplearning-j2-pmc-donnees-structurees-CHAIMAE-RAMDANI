from pathlib import Path
from urllib.error import URLError
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


#  PARAMÈTRES

RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 32

WINE_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

LOCAL_CSV = Path("winequality-red.csv")


# CHARGEMENT DU DATASET

def load_wine_dataset():


    if LOCAL_CSV.exists():
        print(
            f"Chargement depuis le fichier local : "
            f"{LOCAL_CSV}"
        )

        return pd.read_csv(
            LOCAL_CSV,
            sep=";"
        )

    print(
        "Fichier local absent. "
        "Téléchargement depuis l'UCI..."
    )

    try:
        dataframe = pd.read_csv(
            WINE_URL,
            sep=";"
        )

        dataframe.to_csv(
            LOCAL_CSV,
            sep=";",
            index=False
        )

        print(
            f"Dataset sauvegardé dans : "
            f"{LOCAL_CSV}"
        )

        return dataframe

    except (URLError, OSError) as error:
        raise RuntimeError(
            "Impossible de télécharger le dataset. "
            "Télécharge winequality-red.csv et place-le "
            "dans le même dossier que ce script."
        ) from error


df_wine = load_wine_dataset()


print("\n=== WINE QUALITY ROUGE ===")

print(
    "Shape du dataset :",
    df_wine.shape
)

print(
    "\nColonnes :"
)

print(
    df_wine.columns.tolist()
)

print(
    "\nPremières lignes :"
)

print(
    df_wine.head()
)


# VÉRIFICATIONS DU DATASET

expected_columns = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality"
]

missing_columns = [
    column
    for column in expected_columns
    if column not in df_wine.columns
]

if missing_columns:
    raise ValueError(
        "Colonnes manquantes dans le dataset : "
        f"{missing_columns}"
    )


print(
    "\n=== VALEURS MANQUANTES ==="
)

print(
    df_wine.isnull().sum()
)

if df_wine.isnull().any().any():
    raise ValueError(
        "Le dataset contient des valeurs manquantes."
    )


# DISTRIBUTION DES QUALITÉS BRUTES

raw_distribution = (
    df_wine["quality"]
    .value_counts()
    .sort_index()
)

print(
    "\n=== DISTRIBUTION DES QUALITÉS BRUTES ==="
)

for quality, count in raw_distribution.items():
    percentage = (
        count / len(df_wine)
    ) * 100

    print(
        f"Qualité {quality} : "
        f"{count:4d} exemples "
        f"({percentage:5.1f} %)"
    )


#  AGRÉGATION EN TROIS CLASSES

def map_quality(quality):
    """
    Convertit la note brute en 3 classes.

    0 : low
    1 : medium
    2 : high
    """

    if quality <= 4:
        return 0

    if quality <= 6:
        return 1

    return 2


df_wine["quality_3class"] = (
    df_wine["quality"]
    .apply(map_quality)
)


class_names = [
    "low",
    "medium",
    "high"
]


aggregated_distribution = (
    df_wine["quality_3class"]
    .value_counts()
    .sort_index()
)


print(
    "\n=== DISTRIBUTION AGRÉGÉE ==="
)

for class_id, count in (
    aggregated_distribution.items()
):
    percentage = (
        count / len(df_wine)
    ) * 100

    print(
        f"Classe {class_id} "
        f"({class_names[class_id]:6s}) : "
        f"{count:4d} exemples "
        f"({percentage:5.1f} %)"
    )



#  BASELINE NAÏVE

majority_class = (
    aggregated_distribution.idxmax()
)

majority_accuracy = (
    aggregated_distribution.max()
    / len(df_wine)
)


print(
    "\n=== BASELINE NAÏVE ==="
)

print(
    f"Classe majoritaire : "
    f"{majority_class} "
    f"({class_names[majority_class]})"
)

print(
    f"Accuracy si on prédit toujours "
    f"'{class_names[majority_class]}' : "
    f"{majority_accuracy:.4f}"
)

print(
    "Attention : cette accuracy peut être élevée "
    "sans jamais détecter les classes low et high."
)


# SÉPARATION DES FEATURES ET DE LA CIBLE

X_wine = df_wine.drop(
    columns=[
        "quality",
        "quality_3class"
    ]
).to_numpy(
    dtype=np.float32
)

y_wine = df_wine[
    "quality_3class"
].to_numpy(
    dtype=np.int32
)


print(
    "\n=== DONNÉES ==="
)

print(
    "Shape X :",
    X_wine.shape
)

print(
    "Shape y :",
    y_wine.shape
)

print(
    "Classes présentes :",
    np.unique(y_wine)
)


#  SPLIT TRAIN / TEST AVEC STRATIFY

X_train_val, X_test, y_train_val, y_test = (
    train_test_split(
        X_wine,
        y_wine,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_wine
    )
)


# SPLIT TRAIN / VALIDATION AVEC STRATIFY

X_train, X_val, y_train, y_val = (
    train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )
)


print(
    "\n=== SHAPES ==="
)

print(
    "Train      :",
    X_train.shape,
    y_train.shape
)

print(
    "Validation :",
    X_val.shape,
    y_val.shape
)

print(
    "Test       :",
    X_test.shape,
    y_test.shape
)


#  DISTRIBUTION DES CLASSES PAR SPLIT
def print_split_distribution(name, labels):
 

    print(
        f"\nDistribution {name} :"
    )

    values, counts = np.unique(
        labels,
        return_counts=True
    )

    for value, count in zip(
        values,
        counts
    ):
        percentage = (
            count / len(labels)
        ) * 100

        print(
            f"- {class_names[value]:6s} : "
            f"{count:4d} "
            f"({percentage:5.1f} %)"
        )


print_split_distribution(
    "train",
    y_train
)

print_split_distribution(
    "validation",
    y_val
)

print_split_distribution(
    "test",
    y_test
)


# 11. NORMALISATION

scaler = StandardScaler()

X_train_norm = scaler.fit_transform(
    X_train
)

X_val_norm = scaler.transform(
    X_val
)

X_test_norm = scaler.transform(
    X_test
)


print(
    "\n=== NORMALISATION ==="
)

print(
    "Moyennes du train normalisé :"
)

print(
    np.round(
        X_train_norm.mean(axis=0),
        3
    )
)

print(
    "Écarts-types du train normalisé :"
)

print(
    np.round(
        X_train_norm.std(axis=0),
        3
    )
)


# CLASS WEIGHTS


unique_classes = np.unique(
    y_train
)

balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train
)

class_weight = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        unique_classes,
        balanced_weights
    )
}


print(
    "\n=== CLASS WEIGHTS ==="
)

print(
    class_weight
)


# CONSTRUCTION DU MODÈLE BASELINE

def build_wine_baseline(input_dim):
  

    model = keras.Sequential([
        keras.Input(
            shape=(input_dim,)
        ),

        keras.layers.Dense(
            64,
            activation="relu"
        ),

        keras.layers.Dense(
            32,
            activation="relu"
        ),

        keras.layers.Dense(
            3,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss=(
            "sparse_categorical_crossentropy"
        ),

        metrics=[
            "accuracy"
        ]
    )

    return model


keras.backend.clear_session()

keras.utils.set_random_seed(
    RANDOM_STATE
)

model = build_wine_baseline(
    input_dim=X_train_norm.shape[1]
)

model.summary()


#EARLY STOPPING

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)


#  ENTRAÎNEMENT

print(
    "\n=== ENTRAÎNEMENT DU MODÈLE ==="
)

history = model.fit(
    X_train_norm,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(
        X_val_norm,
        y_val
    ),
    callbacks=[
        early_stopping
    ],
    class_weight=class_weight,
    verbose=1
)


#  INFORMATIONS SUR L'ENTRAÎNEMENT

epochs_reels = len(
    history.history["loss"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_accuracy_epoch = (
    int(
        np.argmax(
            history.history[
                "val_accuracy"
            ]
        )
    )
    + 1
)

best_val_loss = min(
    history.history["val_loss"]
)

best_val_loss_epoch = (
    int(
        np.argmin(
            history.history[
                "val_loss"
            ]
        )
    )
    + 1
)


print(
    "\n=== VALIDATION ==="
)

print(
    f"Epochs réellement exécutées : "
    f"{epochs_reels}"
)

print(
    f"Meilleure val_accuracy : "
    f"{best_val_accuracy:.4f} "
    f"(epoch {best_val_accuracy_epoch})"
)

print(
    f"Meilleure val_loss : "
    f"{best_val_loss:.4f} "
    f"(epoch {best_val_loss_epoch})"
)


# ÉVALUATION SUR LE TEST

test_loss, test_accuracy = model.evaluate(
    X_test_norm,
    y_test,
    verbose=0
)


test_probabilities = model.predict(
    X_test_norm,
    verbose=0
)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)


print(
    "\n=== TEST ==="
)

print(
    f"Test loss : "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy : "
    f"{test_accuracy:.4f}"
)


# DISTRIBUTION DES PRÉDICTIONS

prediction_values, prediction_counts = (
    np.unique(
        test_predictions,
        return_counts=True
    )
)


print(
    "\n=== DISTRIBUTION DES PRÉDICTIONS ==="
)

for class_id in range(3):
    count = 0

    if class_id in prediction_values:
        position = np.where(
            prediction_values == class_id
        )[0][0]

        count = prediction_counts[position]

    percentage = (
        count / len(test_predictions)
    ) * 100

    print(
        f"Classe {class_id} "
        f"({class_names[class_id]:6s}) : "
        f"{count:4d} prédictions "
        f"({percentage:5.1f} %)"
    )


if len(np.unique(test_predictions)) < 3:
    print(
        "\nATTENTION : le modèle ne prédit pas "
        "les trois classes."
    )

else:
    print(
        "\nLe modèle prédit bien les trois classes."
    )



# RAPPORT DE CLASSIFICATION

print(
    "\n=== RAPPORT DE CLASSIFICATION ==="
)

print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1, 2],
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# MATRICE DE CONFUSION

confusion = confusion_matrix(
    y_test,
    test_predictions,
    labels=[
        0,
        1,
        2
    ]
)


print(
    "\n=== MATRICE DE CONFUSION ==="
)

print(
    confusion
)


#  QUELQUES PRÉDICTIONS

print(
    "\n=== EXEMPLES DE PRÉDICTIONS ==="
)

for index in range(
    min(10, len(y_test))
):
    probability_text = np.round(
        test_probabilities[index],
        3
    )

    print(
        f"Exemple {index + 1:2d} | "
        f"Réel : "
        f"{class_names[y_test[index]]:6s} | "
        f"Prédit : "
        f"{class_names[test_predictions[index]]:6s} | "
        f"Probabilités : "
        f"{probability_text}"
    )


# GRAPHIQUES

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


# Loss

axes[0, 0].plot(
    history.history["loss"],
    label="Train"
)

axes[0, 0].plot(
    history.history["val_loss"],
    label="Validation"
)

axes[0, 0].set_title(
    "Loss — Wine Quality"
)

axes[0, 0].set_xlabel(
    "Epoch"
)

axes[0, 0].set_ylabel(
    "Sparse categorical cross-entropy"
)

axes[0, 0].legend()

axes[0, 0].grid()


# Accuracy

axes[0, 1].plot(
    history.history["accuracy"],
    label="Train"
)

axes[0, 1].plot(
    history.history["val_accuracy"],
    label="Validation"
)

axes[0, 1].set_title(
    "Accuracy — Wine Quality"
)

axes[0, 1].set_xlabel(
    "Epoch"
)

axes[0, 1].set_ylabel(
    "Accuracy"
)

axes[0, 1].legend()

axes[0, 1].grid()


# Distribution réelle

real_counts = np.bincount(
    y_test,
    minlength=3
)

axes[1, 0].bar(
    class_names,
    real_counts
)

axes[1, 0].set_title(
    "Distribution réelle — Test"
)

axes[1, 0].set_xlabel(
    "Classe"
)

axes[1, 0].set_ylabel(
    "Nombre d'exemples"
)


# Distribution prédite

predicted_counts = np.bincount(
    test_predictions,
    minlength=3
)

axes[1, 1].bar(
    class_names,
    predicted_counts
)

axes[1, 1].set_title(
    "Distribution prédite — Test"
)

axes[1, 1].set_xlabel(
    "Classe"
)

axes[1, 1].set_ylabel(
    "Nombre de prédictions"
)


plt.tight_layout()

plt.savefig(
    "phase7_wine_baseline.png",
    dpi=120,
    bbox_inches="tight"
)

plt.close()


#  IMAGE DE LA MATRICE DE CONFUSION

figure_cm, axis_cm = plt.subplots(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=class_names
)

display.plot(
    ax=axis_cm,
    values_format="d"
)

axis_cm.set_title(
    "Matrice de confusion — Wine Quality"
)

figure_cm.tight_layout()

figure_cm.savefig(
    "phase7_wine_confusion_matrix.png",
    dpi=120,
    bbox_inches="tight"
)

plt.close(
    figure_cm
)


# RÉSUMÉ FINAL

print(
    "\n=== RÉSUMÉ FINAL ==="
)

print(
    f"Baseline naïve classe medium : "
    f"{majority_accuracy:.4f}"
)

print(
    f"Meilleure val_accuracy : "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Test accuracy : "
    f"{test_accuracy:.4f}"
)

print(
    "Images créées :"
)

print(
    "- phase7_wine_baseline.png"
)

print(
    "- phase7_wine_confusion_matrix.png"
)