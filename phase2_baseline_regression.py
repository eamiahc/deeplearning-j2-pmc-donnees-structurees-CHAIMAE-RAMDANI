import numpy as np
import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. CHARGEMENT DES DONNÉES

data = fetch_california_housing()

X = data.data
y = data.target

print("Shape X :", X.shape)
print("Shape y :", y.shape)


# 2. SPLIT TRAIN / VALIDATION / TEST :  80 % train + validation, 20 % test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Parmi les 80 %, on réserve 20 % pour la validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42
)


# NORMALISATION

scaler = StandardScaler()

# Le scaler apprend uniquement sur le train
X_train_norm = scaler.fit_transform(X_train)

# Validation et test utilisent le même scaler
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

print("\n=== SHAPES ===")
print("Train      :", X_train_norm.shape)
print("Validation :", X_val_norm.shape)
print("Test       :", X_test_norm.shape)


# CONSTRUCTION DU MODÈLE

def build_regression_model(input_dim):
    """
    Construit un PMC de régression.

    La couche de sortie ne doit pas utiliser sigmoid :
    le prix est une valeur continue qui peut dépasser 1.
    Une sigmoid limiterait les prédictions entre 0 et 1.
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

        # Régression : un neurone sans activation
        keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# CRÉATION ET RÉSUMÉ

model = build_regression_model(
    input_dim=X_train_norm.shape[1]
)

model.summary()


# ENTRAÎNEMENT

history = model.fit(
    X_train_norm,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(
        X_val_norm,
        y_val
    ),
    verbose=1
)


# ÉVALUATION SUR LE TEST

test_loss, test_mae = model.evaluate(
    X_test_norm,
    y_test,
    verbose=0
)

print("\n=== RÉSULTATS TEST ===")

print(
    f"MSE test : {test_loss:.4f}"
)

print(
    f"MAE test : {test_mae:.4f} "
    "(en centaines de milliers de $)"
)

print(
    f"Erreur moyenne approximative : "
    f"{test_mae * 100000:,.0f} $"
)


#  QUELQUES PRÉDICTIONS

predictions = model.predict(
    X_test_norm[:5],
    verbose=0
).flatten()

print("\n=== EXEMPLES DE PRÉDICTIONS ===")

for index in range(5):
    print(
        f"Exemple {index + 1} | "
        f"Prix réel : {y_test[index]:.3f} | "
        f"Prix prédit : {predictions[index]:.3f}"
    )


# COURBES D'APPRENTISSAGE

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)


# Courbe MSE
axes[0].plot(
    history.history["loss"],
    label="Train"
)

axes[0].plot(
    history.history["val_loss"],
    label="Validation"
)

axes[0].set_title("MSE — California Housing")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid()


# Courbe MAE
axes[1].plot(
    history.history["mae"],
    label="Train"
)

axes[1].plot(
    history.history["val_mae"],
    label="Validation"
)

axes[1].set_title("MAE — California Housing")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].legend()
axes[1].grid()


plt.tight_layout()

plt.savefig(
    "phase2_baseline_regression.png",
    dpi=120,
    bbox_inches="tight"
)

plt.close()

print(
    "\nCourbes enregistrées dans "
    "phase2_baseline_regression.png"
)