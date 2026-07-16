import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Chargement du dataset California Housing
data = fetch_california_housing()

X = data.data
y = data.target



# Informations sur le dataset
print("=== CALIFORNIA HOUSING ===")

print(
    f"Nombre d'exemples : {X.shape[0]}"
)

print(
    f"Nombre de features : {X.shape[1]}"
)

print(
    f"Shape X : {X.shape}"
)

print(
    f"Shape y : {y.shape}"
)


print(
    "\nFeatures :"
)

for feature in data.feature_names:
    print(
        f"- {feature}"
    )


# Statistiques avant normalisation
print(
    "\n=== STATISTIQUES AVANT NORMALISATION ==="
)

print(
    "Moyennes :"
)

print(
    np.round(
        X.mean(axis=0),
        3
    )
)

print(
    "Écarts-types :"
)

print(
    np.round(
        X.std(axis=0),
        3
    )
)


# Split train / test : 80 % train et 20 % test
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)


# Split train / validation : 20 % du train pour la validation
X_train, X_val, y_train, y_val = (
    train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42
    )
)


# Normalisation : fit uniquement sur le TRAIN
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


# Vérification des shapes
print(
    "\n=== SHAPES ==="
)

print(
    f"Train X : {X_train_norm.shape}"
)

print(
    f"Train y : {y_train.shape}"
)

print(
    f"Validation X : {X_val_norm.shape}"
)

print(
    f"Validation y : {y_val.shape}"
)

print(
    f"Test X : {X_test_norm.shape}"
)

print(
    f"Test y : {y_test.shape}"
)


# Vérification de la normalisation
print(
    "\n=== APRÈS NORMALISATION ==="
)

print(
    "Moyennes train :"
)

print(
    np.round(
        X_train_norm.mean(axis=0),
        3
    )
)

print(
    "Écarts-types train :"
)

print(
    np.round(
        X_train_norm.std(axis=0),
        3
    )
)

# Vérifications automatiques
assert X_train_norm.shape[1] == 8

assert X_val_norm.shape[1] == 8

assert X_test_norm.shape[1] == 8


assert np.allclose(
    X_train_norm.mean(axis=0),
    0,
    atol=1e-7
)


assert np.allclose(
    X_train_norm.std(axis=0),
    1,
    atol=1e-7
)


print(
    "\nToutes les vérifications sont validées."
)