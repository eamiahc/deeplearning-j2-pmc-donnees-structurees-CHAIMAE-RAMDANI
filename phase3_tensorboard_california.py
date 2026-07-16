from pathlib import Path
import datetime
import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# CHARGEMENT DES DONNÉES

data = fetch_california_housing()

X = data.data
y = data.target


# SPLIT TRAIN / VALIDATION / TEST

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42
)


#  NORMALISATION

scaler = StandardScaler()

X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)


print("=== SHAPES ===")
print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)


# CONSTRUCTION DU MODÈLE

def build_regression_model(input_dim):
    

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

        keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


#  FONCTION D'ENTRAÎNEMENT AVEC TENSORBOARD

def train_with_tensorboard(
    X_train_data,
    y_train_data,
    X_val_data,
    y_val_data,
    run_name,
    epochs=100
):


    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )

    # Chemin court, sans accent et indépendant de OneDrive
    base_log_dir = Path(
        r"C:\tb_logs\jour2_california"
    )

    base_log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_dir = base_log_dir / (
        f"{run_name}_{timestamp}"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    print(
        f"\nDébut du run : {run_name}"
    )

    print(
        f"Logs enregistrés dans : {log_dir}"
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch"
    )

    model = build_regression_model(
        input_dim=X_train_data.shape[1]
    )

    history = model.fit(
        X_train_data,
        y_train_data,
        epochs=epochs,
        batch_size=32,
        validation_data=(
            X_val_data,
            y_val_data
        ),
        callbacks=[
            tensorboard_callback
        ],
        verbose=0
    )

    print(
        f"Run '{run_name}' terminé."
    )

    print(
        f"Loss finale train : "
        f"{history.history['loss'][-1]:.4f}"
    )

    print(
        f"Loss finale validation : "
        f"{history.history['val_loss'][-1]:.4f}"
    )

    print(
        f"MAE finale train : "
        f"{history.history['mae'][-1]:.4f}"
    )

    print(
        f"MAE finale validation : "
        f"{history.history['val_mae'][-1]:.4f}"
    )

    return model, history, log_dir


#  RUN 1 : DONNÉES NORMALISÉES

model_norm, history_norm, log_norm = train_with_tensorboard(
    X_train_norm,
    y_train,
    X_val_norm,
    y_val,
    run_name="california_norm",
    epochs=100
)


#  RUN 2 : DONNÉES BRUTES

model_raw, history_raw, log_raw = train_with_tensorboard(
    X_train,
    y_train,
    X_val,
    y_val,
    run_name="california_raw",
    epochs=100
)


# ÉVALUATION SUR LE TEST

test_loss_norm, test_mae_norm = model_norm.evaluate(
    X_test_norm,
    y_test,
    verbose=0
)

test_loss_raw, test_mae_raw = model_raw.evaluate(
    X_test,
    y_test,
    verbose=0
)


# COMPARAISON FINALE

print(
    "\n=== COMPARAISON DES DEUX RUNS ==="
)

print(
    "\nDonnées normalisées :"
)

print(
    f"- Train loss finale : "
    f"{history_norm.history['loss'][-1]:.4f}"
)

print(
    f"- Val loss finale   : "
    f"{history_norm.history['val_loss'][-1]:.4f}"
)

print(
    f"- Test loss         : "
    f"{test_loss_norm:.4f}"
)

print(
    f"- Test MAE          : "
    f"{test_mae_norm:.4f}"
)


print(
    "\nDonnées brutes :"
)

print(
    f"- Train loss finale : "
    f"{history_raw.history['loss'][-1]:.4f}"
)

print(
    f"- Val loss finale   : "
    f"{history_raw.history['val_loss'][-1]:.4f}"
)

print(
    f"- Test loss         : "
    f"{test_loss_raw:.4f}"
)

print(
    f"- Test MAE          : "
    f"{test_mae_raw:.4f}"
)


# EMPLACEMENT DES LOGS

print(
    "\n=== DOSSIERS TENSORBOARD ==="
)

print(
    f"Run normalisé : {log_norm}"
)

print(
    f"Run brut      : {log_raw}"
)

print(
    "\nPour lancer TensorBoard :"
)

print(
    r'tensorboard --logdir="C:\tb_logs\jour2_california"'
)

print(
    "\nPuis ouvrir : http://localhost:6006"
)