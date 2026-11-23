"""Experimento de features y validación para el modelo de inasistencias.

Compara el conjunto de features del notebook 05 contra un conjunto ampliado
con variables temporales e historial causal del paciente, usando el mismo
GroupKFold(5) por PatientId para que la comparación sea justa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

RUTA_DATOS = "data/citas_medicas_preprocesadas.csv"
SEMILLA = 42


def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(RUTA_DATOS)
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
    for col in ("ScheduledDay", "AppointmentDay"):
        if df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)
    df["No-show"] = df["No-show"].astype(int)
    return df


def agregar_historial_causal(df: pd.DataFrame, alpha: float = 20.0, modo: str = "agendamiento") -> pd.DataFrame:
    """Agrega historia del paciente usando solo citas anteriores.

    modo="agendamiento": solo citas atendidas antes de la fecha de agendamiento.
    modo="cita": citas atendidas antes de la cita actual (scoring cercano a la cita).
    """
    d = df.sort_values(["PatientId", "ScheduledDay", "AppointmentDay"]).reset_index(drop=True)
    n = len(d)
    tasa_global = float(d["No-show"].mean())

    conteo_previo = np.zeros(n, dtype=np.int32)
    faltas_previas = np.zeros(n, dtype=np.int32)
    ultima_falta = np.full(n, np.nan, dtype=np.float64)
    dias_desde_ultima = np.full(n, np.nan, dtype=np.float64)

    for _, idx in d.groupby("PatientId", sort=False).indices.items():
        idx = np.asarray(idx)
        citas = d.loc[idx, "AppointmentDay"].to_numpy(dtype="datetime64[D]")
        agendamientos = d.loc[idx, "ScheduledDay"].to_numpy(dtype="datetime64[D]")
        objetivos = d.loc[idx, "No-show"].to_numpy(dtype=np.int8)

        orden = np.argsort(citas, kind="stable")
        citas_ordenadas = citas[orden]
        objetivos_ordenados = objetivos[orden]
        acumulado = np.concatenate(([0], np.cumsum(objetivos_ordenados)))

        referencia = agendamientos if modo == "agendamiento" else citas
        previas = np.searchsorted(citas_ordenadas, referencia, side="left")
        conteo_previo[idx] = previas
        faltas_previas[idx] = acumulado[previas]

        con_historia = previas > 0
        if con_historia.any():
            posiciones = previas[con_historia] - 1
            ultima_falta[idx[con_historia]] = objetivos_ordenados[posiciones]
            dias_desde_ultima[idx[con_historia]] = (
                referencia[con_historia] - citas_ordenadas[posiciones]
            ).astype(np.float64)

    d["PriorVisits"] = conteo_previo
    d["PriorNoShows"] = faltas_previas
    d["PriorNoShowRate"] = (faltas_previas + alpha * tasa_global) / (conteo_previo + alpha)
    d["HasHistory"] = (conteo_previo > 0).astype(np.int8)
    d["DaysSinceLastVisit"] = dias_desde_ultima
    d["LastVisitNoShow"] = ultima_falta
    return d


def construir_features(df: pd.DataFrame, modo_historial: str = "agendamiento") -> pd.DataFrame:
    d = df.copy()
    # Nombres originales del notebook 05 para poder reproducir su baseline.
    d["ScheduledDay_month"] = d["ScheduledDay"].dt.month
    d["ScheduledDay_day"] = d["ScheduledDay"].dt.day
    d["ScheduledDay_hour"] = d["ScheduledDay"].dt.hour
    d["AppointmentDay_month"] = d["AppointmentDay"].dt.month
    d["AppointmentDay_day"] = d["AppointmentDay"].dt.day
    # Variables nuevas.
    d["ScheduledDayOfWeek"] = d["ScheduledDay"].dt.day_name()
    d["ScheduledHour"] = d["ScheduledDay"].dt.hour
    d["ScheduledMonth"] = d["ScheduledDay"].dt.month
    d["ScheduledDayOfMonth"] = d["ScheduledDay"].dt.day
    d["AppointmentWeek"] = d["AppointmentDay"].dt.isocalendar().week.astype(int)
    d["AppointmentDayOfMonth"] = d["AppointmentDay"].dt.day
    d["AppointmentMonth"] = d["AppointmentDay"].dt.month
    inicio = d["AppointmentDay"].min().normalize()
    d["DaysSinceStart"] = (d["AppointmentDay"].dt.normalize() - inicio).dt.days
    d["IsSameDay"] = (d["WaitingDays"] == 0).astype(np.int8)
    d["WaitingDaysLog"] = np.log1p(d["WaitingDays"])
    d["HasHandcap"] = (d["Handcap"] > 0).astype(np.int8)
    d["IsChild"] = (d["Age"] < 18).astype(np.int8)
    d["IsSenior"] = (d["Age"] >= 60).astype(np.int8)
    d["AgeGroup"] = pd.cut(
        d["Age"],
        bins=[-1, 0, 12, 17, 29, 44, 59, 74, 120],
        labels=["0", "1-12", "13-17", "18-29", "30-44", "45-59", "60-74", "75+"],
    ).astype(str)
    d["ChronicAndSenior"] = d["IsSenior"] * d["Num_Conditions"]
    d = agregar_historial_causal(d, modo=modo_historial)
    return d


BASE_NUMERICAS = [
    "Age",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
    "WaitingDays",
    "Num_Conditions",
    "ScheduledDay_month",
    "ScheduledDay_day",
    "ScheduledDay_hour",
    "AppointmentDay_month",
    "AppointmentDay_day",
]
BASE_CATEGORICAS = ["Gender", "Neighbourhood", "AppointmentDayOfWeek"]

MEJOR_NUMERICAS = BASE_NUMERICAS + [
    "ScheduledHour",
    "ScheduledMonth",
    "ScheduledDayOfMonth",
    "AppointmentWeek",
    "AppointmentDayOfMonth",
    "AppointmentMonth",
    "DaysSinceStart",
    "IsSameDay",
    "WaitingDaysLog",
    "HasHandcap",
    "IsChild",
    "IsSenior",
    "ChronicAndSenior",
    "PriorVisits",
    "PriorNoShows",
    "PriorNoShowRate",
    "HasHistory",
    "DaysSinceLastVisit",
    "LastVisitNoShow",
]
MEJOR_CATEGORICAS = BASE_CATEGORICAS + ["ScheduledDayOfWeek", "AgeGroup"]


@dataclass
class Resultado:
    nombre: str
    roc_auc: float
    pr_auc: float
    umbral: float
    exactitud: float
    precision: float
    recall: float
    f1: float
    lift20: float
    recall20: float


def metricas(y_true: np.ndarray, probas: np.ndarray) -> Resultado:
    from sklearn.metrics import precision_recall_curve

    precisiones, recalls, umbrales = precision_recall_curve(y_true, probas)
    f1s = 2 * precisiones * recalls / (precisiones + recalls + 1e-10)
    mejor = int(np.argmax(f1s))
    umbral = float(umbrales[mejor]) if mejor < len(umbrales) else 0.5
    pred = (probas >= umbral).astype(int)

    orden = np.argsort(probas)[::-1]
    k = max(1, int(round(0.2 * len(y_true))))
    tasa_global = float(y_true.mean())
    tasa_top = float(y_true[orden[:k]].mean())
    lift20 = tasa_top / tasa_global if tasa_global > 0 else float("nan")
    recall20 = float(y_true[orden[:k]].sum() / max(1, y_true.sum()))

    return Resultado(
        nombre="",
        roc_auc=float(roc_auc_score(y_true, probas)),
        pr_auc=float(average_precision_score(y_true, probas)),
        umbral=umbral,
        exactitud=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        lift20=lift20,
        recall20=recall20,
    )


def evaluar(
    nombre: str,
    df: pd.DataFrame,
    numericas: list[str],
    categoricas: list[str],
    params: dict | None = None,
) -> tuple[Resultado, np.ndarray]:
    X = df[numericas + categoricas].copy()
    for col in categoricas:
        X[col] = X[col].astype("category")
    y = df["No-show"].to_numpy(dtype=int)
    grupos = df["PatientId"].to_numpy()

    params = params or {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "is_unbalance": True,
        "random_state": SEMILLA,
        "verbose": -1,
        "n_jobs": 8,
    }

    oof = np.zeros(len(df))
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=grupos), start=1):
        modelo = LGBMClassifier(**params)
        modelo.fit(X.iloc[tr], y[tr])
        oof[va] = modelo.predict_proba(X.iloc[va])[:, 1]
        print(f"  {nombre}: fold {fold}/5 listo", flush=True)

    resultado = metricas(y, oof)
    resultado.nombre = nombre
    return resultado, oof


def imprimir(resultado: Resultado) -> None:
    print(
        f"{resultado.nombre:10s} | ROC-AUC {resultado.roc_auc:.4f} | "
        f"PR-AUC {resultado.pr_auc:.4f} | F1 {resultado.f1:.4f} | "
        f"recall {resultado.recall:.4f} | prec {resultado.precision:.4f} | "
        f"umbral {resultado.umbral:.4f} | Lift@20 {resultado.lift20:.4f} | "
        f"recall@20 {resultado.recall20:.4f}"
    )


if __name__ == "__main__":
    df_original = cargar_datos()

    df = construir_features(df_original, modo_historial="agendamiento")
    print(f"Datos: {df.shape}, tasa No-show: {df['No-show'].mean():.4f}")
    print(f"Pacientes: {df['PatientId'].nunique()}, con historial: {df['HasHistory'].mean():.3f}")

    print("\n=== Baseline notebook 05 ===")
    base, _ = evaluar("baseline", df, BASE_NUMERICAS, BASE_CATEGORICAS)
    imprimir(base)

    print("\n=== Features ampliadas (historial al agendar) ===")
    mejor, _ = evaluar("ampliado", df, MEJOR_NUMERICAS, MEJOR_CATEGORICAS)
    imprimir(mejor)

    df_cita = construir_features(df_original, modo_historial="cita")
    print("\n=== Features ampliadas (historial al momento de la cita) ===")
    mejor_cita, _ = evaluar("hist_cita", df_cita, MEJOR_NUMERICAS, MEJOR_CATEGORICAS)
    imprimir(mejor_cita)
