"""Búsqueda de hiperparámetros para el modelo LightGBM mejorado."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from experimento_modelo import (
    MEJOR_CATEGORICAS,
    MEJOR_NUMERICAS,
    cargar_datos,
    construir_features,
    evaluar,
    imprimir,
)

SEMILLA = 42
RUTA_MODELOS = Path("models")
RUTA_MODELOS.mkdir(exist_ok=True)


def preparar_matriz(df):
    X = df[MEJOR_NUMERICAS + MEJOR_CATEGORICAS].copy()
    for col in MEJOR_CATEGORICAS:
        X[col] = X[col].astype("category")
    y = df["No-show"].to_numpy(dtype=int)
    grupos = df["PatientId"].to_numpy()
    return X, y, grupos


def evaluar_cv(params: dict, X, y, grupos, n_splits: int = 3):
    prs, aucs = [], []
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups=grupos):
        modelo = LGBMClassifier(**params)
        modelo.fit(X.iloc[tr], y[tr])
        probas = modelo.predict_proba(X.iloc[va])[:, 1]
        prs.append(average_precision_score(y[va], probas))
        aucs.append(roc_auc_score(y[va], probas))
    return float(np.mean(prs)), float(np.mean(aucs)), int(params["n_estimators"])


def objective(trial: optuna.Trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1400, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.05),
        "is_unbalance": False,
        "random_state": SEMILLA,
        "verbose": -1,
        "n_jobs": 8,
    }
    pr_auc, roc_auc, _ = evaluar_cv(params, X, y, grupos, n_splits=3)
    trial.set_user_attr("roc_auc", roc_auc)
    print(f"Trial {trial.number}: PR-AUC={pr_auc:.4f} ROC-AUC={roc_auc:.4f}", flush=True)
    return pr_auc


if __name__ == "__main__":
    df = construir_features(cargar_datos(), modo_historial="cita")
    X, y, grupos = preparar_matriz(df)
    print(f"Matriz: {X.shape}, tasa No-show={y.mean():.4f}", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEMILLA),
    )
    estudio.optimize(objective, n_trials=40, timeout=1500, show_progress_bar=False)

    print("\n--- Mejor configuración ---")
    print(f"Mejor PR-AUC CV(3): {estudio.best_value:.4f}")
    print(f"ROC-AUC CV(3): {estudio.best_trial.user_attrs['roc_auc']:.4f}")
    print(json.dumps(estudio.best_params, indent=2, ensure_ascii=False))

    mejores = dict(estudio.best_params)
    mejores.update(
        {
            "subsample_freq": 1,
            "is_unbalance": False,
            "random_state": SEMILLA,
            "verbose": -1,
            "n_jobs": 8,
        }
    )

    with open(RUTA_MODELOS / "mejores_hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump(mejores, f, indent=2, ensure_ascii=False)

    print("\n=== Evaluación final 5-fold OOF con mejores hiperparámetros ===")
    resultado, _ = evaluar("mejorado", df, MEJOR_NUMERICAS, MEJOR_CATEGORICAS, mejores)
    imprimir(resultado)
