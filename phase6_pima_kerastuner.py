from pathlib import Path
import pandas as pd
import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# CHARGEMENT DU DATASET


csv_path = Path("diabetes.csv")

if not csv_path.exists():
    raise FileNotFoundError(
        "Le fichier diabetes.csv est introuvable. "
        "Place-le dans le même dossier que ce script."
    )

df = pd.read_csv(csv_path)

print("=== PIMA — KERAS TUNER ===")
print("Shape :", df.shape)


# TRAITEMENT DES ZÉROS SUSPECTS

zero_suspect_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

print("\n=== TRAITEMENT DES ZÉROS SUSPECTS ===")

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


# SÉPARATION X / y

X = df.drop(
    columns=["Outcome"]
).copy()

y = df["Outcome"].astype(int).copy()


# SPLIT TRAIN / TEST


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


# NORMALISATION

scaler = StandardScaler()

X_train_norm = scaler.fit_transform(
    X_train
)

X_test_norm = scaler.transform(
    X_test
)


print("\n=== SHAPES ===")
print("Train :", X_train_norm.shape)
print("Test  :", X_test_norm.shape)


# DÉFINITION DE L'HYPERMODEL

def build_pima_model(hp):

    units_1 = hp.Int(
        "units_1",
        min_value=32,
        max_value=128,
        step=32
    )

    units_2 = hp.Int(
        "units_2",
        min_value=16,
        max_value=64,
        step=16
    )

    activation = hp.Choice(
        "activation",
        values=[
            "relu",
            "tanh"
        ]
    )

    dropout_rate = hp.Float(
        "dropout_rate",
        min_value=0.0,
        max_value=0.5,
        step=0.1
    )

    learning_rate = hp.Choice(
        "learning_rate",
        values=[
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2
        ]
    )

    model = keras.Sequential([
        keras.Input(shape=(8,)),

        keras.layers.Dense(
            units_1,
            activation=activation
        )
    ])

    if dropout_rate > 0:
        model.add(
            keras.layers.Dropout(
                dropout_rate
            )
        )

    model.add(
        keras.layers.Dense(
            units_2,
            activation=activation
        )
    )

    if dropout_rate > 0:
        model.add(
            keras.layers.Dropout(
                dropout_rate
            )
        )

    model.add(
        keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


#   DOSSIER DU TUNER SANS ACCENT

tuner_directory = Path(
    r"C:\keras_tuner_logs\jour2_pima"
)

tuner_directory.mkdir(
    parents=True,
    exist_ok=True
)

print(
    "\nDossier Keras Tuner :",
    tuner_directory
)


# CRÉATION DU RANDOM SEARCH

tuner = kt.RandomSearch(
    hypermodel=build_pima_model,
    objective="val_accuracy",
    max_trials=15,
    seed=42,
    directory=str(tuner_directory),
    project_name="pima_random",
    overwrite=True
)


# AFFICHAGE DE L'ESPACE DE RECHERCHE

print("\n=== ESPACE DE RECHERCHE ===")

tuner.search_space_summary()


# 10. EARLY STOPPING POUR LE SEARCH

early_stop_search = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# LANCEMENT DES TRIALS

print("\n=== DÉBUT DU RANDOM SEARCH ===")

tuner.search(
    X_train_norm,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.20,
    callbacks=[
        early_stop_search
    ],
    verbose=1
)


#  RÉSUMÉ DES 5 MEILLEURS TRIALS

print("\n=== TOP 5 DES TRIALS ===")

tuner.results_summary(
    num_trials=5
)


# MEILLEURS HYPERPARAMÈTRES

best_hp = tuner.get_best_hyperparameters(
    num_trials=1
)[0]


print("\n=== MEILLEURE CONFIGURATION ===")

print(
    "learning_rate :",
    best_hp.get("learning_rate")
)

print(
    "units_1 :",
    best_hp.get("units_1")
)

print(
    "units_2 :",
    best_hp.get("units_2")
)

print(
    "activation :",
    best_hp.get("activation")
)

print(
    "dropout_rate :",
    best_hp.get("dropout_rate")
)


#  AFFICHER LES 5 MEILLEURES CONFIGURATIONS

print(
    "\n=== HYPERPARAMÈTRES DES 5 MEILLEURS ==="
)

top_five = tuner.get_best_hyperparameters(
    num_trials=5
)

for index, hp in enumerate(
    top_five,
    start=1
):

    print(
        f"\nConfiguration {index} :"
    )

    print(
        hp.values
    )


# RECONSTRUCTION DU MEILLEUR MODÈLE

best_model = tuner.hypermodel.build(
    best_hp
)


# EARLY STOPPING FINAL

early_stop_final = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)


# ENTRAÎNEMENT FINAL

print(
    "\n=== ENTRAÎNEMENT FINAL DU MEILLEUR MODÈLE ==="
)

history_best = best_model.fit(
    X_train_norm,
    y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.20,
    callbacks=[
        early_stop_final
    ],
    verbose=1
)


#  ÉVALUATION

best_val_accuracy = max(
    history_best.history["val_accuracy"]
)

epochs_reels = len(
    history_best.history["val_accuracy"]
)

test_loss, test_accuracy = (
    best_model.evaluate(
        X_test_norm,
        y_test,
        verbose=0
    )
)


print("\n=== RÉSULTATS FINAUX ===")

print(
    f"Epochs réels : "
    f"{epochs_reels}"
)

print(
    f"Best val_accuracy : "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Test loss : "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy : "
    f"{test_accuracy:.4f}"
)


# VÉRIFICATION DES PRÉDICTIONS

test_probabilities = best_model.predict(
    X_test_norm,
    verbose=0
).flatten()

test_predictions = (
    test_probabilities >= 0.5
).astype(int)


print(
    f"Moyenne des probabilités positives : "
    f"{test_probabilities.mean():.4f}"
)

print(
    f"Proportion de classes prédites 1 : "
    f"{test_predictions.mean():.4f}"
)


# RAPPEL DU DOSSIER

print(
    "\nRésultats Keras Tuner enregistrés dans :"
)

print(
    tuner_directory / "pima_random"
)