from pathlib import Path
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


#  CHARGEMENT DU DATASET

csv_path = Path("diabetes.csv")

if not csv_path.exists():
    raise FileNotFoundError(
        "Le fichier diabetes.csv est introuvable. "
        "Place-le dans le même dossier que ce script."
    )

df = pd.read_csv(csv_path)


print("=== PIMA DIABETES ===")
print("Shape :", df.shape)

print("\nColonnes :")
print(df.columns.tolist())

print("\nPremières lignes :")
print(df.head())


# VÉRIFICATION DES VALEURS MANQUANTES

print("\n=== VALEURS MANQUANTES ===")
print(df.isnull().sum())


#  DISTRIBUTION DES CLASSES

class_counts = df["Outcome"].value_counts().sort_index()
class_percentages = (
    df["Outcome"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\n=== DISTRIBUTION DES CLASSES ===")

for classe in class_counts.index:
    print(
        f"Classe {classe} : "
        f"{class_counts[classe]} "
        f"({class_percentages[classe]:.1f}%)"
    )


# Baseline naïve : toujours prédire la classe majoritaire
majority_baseline = class_counts.max() / len(df)

print(
    f"\nAccuracy d'un modèle qui prédit "
    f"toujours la classe majoritaire : "
    f"{majority_baseline:.4f}"
)


#  ZÉROS SUSPECTS

zero_suspect_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

print("\n=== ZÉROS SUSPECTS ===")

for column in zero_suspect_columns:
    zero_count = (df[column] == 0).sum()

    print(
        f"{column:20s} : {zero_count}"
    )


# SÉPARATION X / y

X = df.drop(
    columns=["Outcome"]
).copy()

y = df["Outcome"].astype(int).copy()


# SPLIT TRAIN / VALIDATION / TEST : 20 % pour le test.
# Puis 20 % du train restant pour la validation.
# stratify conserve la proportion des classes.

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
print("Train      :", X_train.shape, y_train.shape)
print("Validation :", X_val.shape, y_val.shape)
print("Test       :", X_test.shape, y_test.shape)


# NORMALISATIO

scaler = StandardScaler()

X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)


#  CONSTRUCTION DU MODÈLE BASELINE

def build_pima_baseline(input_dim):
    """
    PMC baseline pour classification binaire.
    """

    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),

        keras.layers.Dense(
            64,
            activation="relu"
        ),

        keras.layers.Dense(
            32,
            activation="relu"
        ),

        keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


keras.utils.set_random_seed(42)

model = build_pima_baseline(
    input_dim=X_train_norm.shape[1]
)

model.summary()

#  ENTRAÎNEMENT

history = model.fit(
    X_train_norm,
    y_train,
    epochs=150,
    batch_size=32,
    validation_data=(
        X_val_norm,
        y_val
    ),
    verbose=1
)


#  RÉSULTATS DE VALIDATION

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_epoch = (
    int(
        np.argmax(
            history.history["val_accuracy"]
        )
    )
    + 1
)

print("\n=== VALIDATION ===")

print(
    f"Meilleure val_accuracy : "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Meilleure epoch : "
    f"{best_epoch}"
)


#  ÉVALUATION TEST

test_loss, test_accuracy = model.evaluate(
    X_test_norm,
    y_test,
    verbose=0
)

test_probabilities = model.predict(
    X_test_norm,
    verbose=0
).flatten()

test_predictions = (
    test_probabilities >= 0.5
).astype(int)


print("\n=== TEST ===")

print(
    f"Test loss     : "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy : "
    f"{test_accuracy:.4f}"
)

print(
    f"Moyenne des probabilités positives : "
    f"{test_probabilities.mean():.4f}"
)

print(
    f"Proportion de classes prédites 1 : "
    f"{test_predictions.mean():.4f}"
)


#  MATRICE DE CONFUSION ET RAPPORT

print("\n=== MATRICE DE CONFUSION ===")

cm = confusion_matrix(
    y_test,
    test_predictions
)

print(cm)


print("\n=== RAPPORT DE CLASSIFICATION ===")

print(
    classification_report(
        y_test,
        test_predictions,
        digits=4
    )
)


# COURBES D'APPRENTISSAGE

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)


axes[0].plot(
    history.history["loss"],
    label="Train"
)

axes[0].plot(
    history.history["val_loss"],
    label="Validation"
)

axes[0].set_title(
    "Loss — Pima baseline"
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary cross-entropy")
axes[0].legend()
axes[0].grid()


axes[1].plot(
    history.history["accuracy"],
    label="Train"
)

axes[1].plot(
    history.history["val_accuracy"],
    label="Validation"
)

axes[1].set_title(
    "Accuracy — Pima baseline"
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid()


plt.tight_layout()

plt.savefig(
    "phase4_pima_baseline.png",
    dpi=120,
    bbox_inches="tight"
)

plt.close()


print(
    "\nCourbes enregistrées dans : "
    "phase4_pima_baseline.png"
)