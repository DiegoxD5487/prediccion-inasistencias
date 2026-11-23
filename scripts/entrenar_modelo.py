"""Entrena y exporta el modelo mejorado de predicción de inasistencias.

Genera en models/:
  - mejor_modelo_lgbm.joblib
  - metricas.json
  - README.md

Uso:
    .venv\\Scripts\\python.exe scripts\\entrenar_modelo.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupKFold

from experimento_modelo import (
    BASE_CATEGORICAS,
    BASE_NUMERICAS,
    MEJOR_CATEGORICAS,
    MEJOR_NUMERICAS,
    Resultado,
    cargar_datos,
    construir_features,
    evaluar,
    imprimir,
    metricas,
)

RUTA_MODELOS = Path("models")
RUTA_MODELOS.mkdir(exist_ok=True)
SEMILLA = 42
FECHA_CORTE_TEMPORAL = "2016-05-25"

PARAMS_BASE = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "is_unbalance": True,
    "random_state": SEMILLA,
    "verbose": -1,
    "n_jobs": 8,
}


def cargar_mejores_params() -> dict:
    with open(RUTA_MODELOS / "mejores_hiperparametros.json", encoding="utf-8") as f:
        params = json.load(f)
    params.setdefault("subsample_freq", 1)
    params["n_jobs"] = 8
    return params


def matriz(df: pd.DataFrame, numericas: list[str], categoricas: list[str]):
    X = df[numericas + categoricas].copy()
    for col in categoricas:
        X[col] = X[col].astype("category")
    y = df["No-show"].to_numpy(dtype=int)
    grupos = df["PatientId"].to_numpy()
    return X, y, grupos


def umbrales(y: np.ndarray, probas: np.ndarray) -> dict:
    precisiones, recalls, cortes = precision_recall_curve(y, probas)
    f1s = 2 * precisiones * recalls / (precisiones + recalls + 1e-10)
    idx_f1 = int(np.argmax(f1s))
    corte_f1 = float(cortes[idx_f1]) if idx_f1 < len(cortes) else 0.5

    mascara = recalls[:-1] >= 0.65
    if mascara.any():
        idx_recall = int(np.where(mascara)[0].max())
        corte_recall = float(cortes[idx_recall])
    else:
        idx_recall = idx_f1
        corte_recall = corte_f1

    return {
        "f1_optimo": {
            "umbral": corte_f1,
            "precision": float(precisiones[idx_f1]),
            "recall": float(recalls[idx_f1]),
            "f1": float(f1s[idx_f1]),
        },
        "recall_065": {
            "umbral": corte_recall,
            "precision": float(precisiones[idx_recall]),
            "recall": float(recalls[idx_recall]),
            "f1": float(f1s[idx_recall]),
        },
    }


def evaluar_temporal(df: pd.DataFrame, numericas, categoricas, params: dict) -> Resultado:
    train = df[df["AppointmentDay"] <= FECHA_CORTE_TEMPORAL]
    test = df[df["AppointmentDay"] > FECHA_CORTE_TEMPORAL]
    X_train = train[numericas + categoricas].copy()
    X_test = test[numericas + categoricas].copy()
    for col in categoricas:
        # Usar el mismo mapeo de categorías en train y test; si no, los códigos
        # internos de pandas no coinciden y las predicciones se distorsionan.
        categorias = df[col].astype("category").cat.categories
        X_train[col] = pd.Categorical(X_train[col], categories=categorias)
        X_test[col] = pd.Categorical(X_test[col], categories=categorias)
    y_train = train["No-show"].to_numpy(dtype=int)
    y_test = test["No-show"].to_numpy(dtype=int)
    modelo = LGBMClassifier(**params)
    modelo.fit(X_train, y_train)
    res = metricas(y_test, modelo.predict_proba(X_test)[:, 1])
    res.nombre = "temporal"
    return res


def resumen(res: Resultado) -> dict:
    datos = asdict(res)
    datos.pop("nombre", None)
    return datos


def main() -> None:
    df = construir_features(cargar_datos(), modo_historial="cita")
    print(f"Datos preparados: {df.shape}, tasa No-show={df['No-show'].mean():.4f}")

    print("\n=== Baseline notebook 05 (GroupKFold 5) ===")
    res_base, _ = evaluar(
        "baseline",
        df,
        BASE_NUMERICAS,
        BASE_CATEGORICAS + ["PatientId"],
        PARAMS_BASE,
    )
    imprimir(res_base)

    params_mejor = cargar_mejores_params()
    print("\n=== Modelo mejorado (GroupKFold 5) ===")
    res_mejor, oof_mejor = evaluar(
        "mejorado",
        df,
        MEJOR_NUMERICAS,
        MEJOR_CATEGORICAS,
        params_mejor,
    )
    imprimir(res_mejor)

    print("\n=== Validación temporal secundaria ===")
    res_base_temporal = evaluar_temporal(
        df,
        BASE_NUMERICAS,
        BASE_CATEGORICAS + ["PatientId"],
        PARAMS_BASE,
    )
    res_mejor_temporal = evaluar_temporal(
        df,
        MEJOR_NUMERICAS,
        MEJOR_CATEGORICAS,
        params_mejor,
    )
    imprimir(res_base_temporal)
    imprimir(res_mejor_temporal)

    cortes = umbrales(df["No-show"].to_numpy(dtype=int), oof_mejor)

    # Modelo final: se reentrena con el 100% de los datos usando los
    # hiperparámetros elegidos por validación. El umbral se elige con OOF,
    # nunca con las predicciones de entrenamiento.
    X, y, _ = matriz(df, MEJOR_NUMERICAS, MEJOR_CATEGORICAS)
    modelo_final = LGBMClassifier(**params_mejor)
    modelo_final.fit(X, y)

    categorias = {
        col: [str(valor) for valor in X[col].cat.categories]
        for col in MEJOR_CATEGORICAS
    }
    tasa_global = float(y.mean())

    payload = {
        "model": modelo_final,
        "target": "No-show",
        "positive_class": "No-show=True (1): el paciente no asistió",
        "features": MEJOR_NUMERICAS + MEJOR_CATEGORICAS,
        "numeric_features": MEJOR_NUMERICAS,
        "categorical_features": MEJOR_CATEGORICAS,
        "categorical_categories": categorias,
        "global_no_show_rate": tasa_global,
        "feature_defaults": {
            "PriorVisits": 0,
            "PriorNoShows": 0,
            "PriorNoShowRate": tasa_global,
            "HasHistory": 0,
            "DaysSinceLastVisit": None,
            "LastVisitNoShow": None,
        },
        "history_mode": "cita",
        "threshold": cortes["f1_optimo"]["umbral"],
        "threshold_recall_065": cortes["recall_065"]["umbral"],
        "thresholds": cortes,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_rows": int(len(df)),
        "training_patients": int(df["PatientId"].nunique()),
        "lightgbm_version": __import__("lightgbm").__version__,
        "params": params_mejor,
        "metrics_oof": resumen(res_mejor),
        "baseline_oof": resumen(res_base),
        "temporal": {
            "corte": FECHA_CORTE_TEMPORAL,
            "baseline": resumen(res_base_temporal),
            "mejorado": resumen(res_mejor_temporal),
        },
        "nota": (
            "Umbral recomendado: threshold_recall_065 para cumplir recall>=0.65 "
            "maximizando precision; threshold (F1) para el mejor equilibrio F1. "
            "Las métricas OOF usan GroupKFold(5) por PatientId y son la referencia "
            "honesta; no uses métricas calculadas sobre el entrenamiento. "
            "Al predecir, fija las categorías con categorical_categories."
        ),
    }

    ruta_modelo = RUTA_MODELOS / "mejor_modelo_lgbm.joblib"
    joblib.dump(payload, ruta_modelo)

    metricas_json = {
        "baseline_notebook_05": resumen(res_base),
        "modelo_mejorado": resumen(res_mejor),
        "mejora_absoluta": {
            "roc_auc": res_mejor.roc_auc - res_base.roc_auc,
            "pr_auc": res_mejor.pr_auc - res_base.pr_auc,
            "f1": res_mejor.f1 - res_base.f1,
            "lift20": res_mejor.lift20 - res_base.lift20,
            "recall20": res_mejor.recall20 - res_base.recall20,
        },
        "validacion_temporal": {
            "corte": FECHA_CORTE_TEMPORAL,
            "baseline": resumen(res_base_temporal),
            "modelo_mejorado": resumen(res_mejor_temporal),
        },
        "umbrales": cortes,
        "protocolo": "GroupKFold(5) agrupado por PatientId; umbral elegido con OOF.",
        "generado_en": payload["trained_at"],
    }
    with open(RUTA_MODELOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas_json, f, indent=2, ensure_ascii=False)

    print(f"\nModelo exportado: {ruta_modelo}")
    print(f"Umbral F1: {cortes['f1_optimo']['umbral']:.4f}")
    print(f"Umbral recall>=0.65: {cortes['recall_065']['umbral']:.4f}")


if __name__ == "__main__":
    main()
