#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelo_multisalida_euros.py
- Carga dataset_sin_total.csv (intenta utf-8, latin1, cp1252)
- Carga modelo multisalida si existe; si no, entrena y guarda
- Calcula R2 y MSE por ciudad e imprime R2
- Muestra y guarda gráficas con ejes formateados en euros
- Guarda outputs en carpeta 'outputs/'
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------- Configuración ----------
CSV_PATH = r"C:\Users\fredd\Desktop\CURSO ATU #2\DATA SET.csv"   # Ajusta si tu CSV está en otra ruta
MODEL_PATH = "modelo_multisalida.pkl"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_COLS = [
    "superficie_m2", "habitaciones",
    "precio_m2_madrid", "precio_m2_barcelona", "precio_m2_valencia",
    "precio_m2_sevilla", "precio_m2_bilbao", "precio_m2_malaga", "precio_m2_jaen"
]

Y_COLS = [
    "presupuesto_madrid", "presupuesto_barcelona", "presupuesto_valencia",
    "presupuesto_sevilla", "presupuesto_bilbao", "presupuesto_malaga",
    "presupuesto_jaen"
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 250

# ---------- Utilidades ----------
def cargar_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra el CSV en: {path}")
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"CSV cargado con encoding: {enc}")
            return df
        except Exception:
            continue
    raise UnicodeError("No se pudo leer el CSV con encodings comunes (utf-8, latin1, cp1252).")

def comprobar_columnas(df):
    faltantes = [c for c in X_COLS + Y_COLS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {faltantes}")

def entrenar_modelo(X_train, y_train):
    base = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model = MultiOutputRegressor(base)
    print("Entrenando modelo multisalida (Random Forest)... esto puede tardar unos segundos.")
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")
    return model

def guardar_modelo(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en: {path}")

def cargar_modelo(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Modelo cargado desde: {path}")
    return model

def formatear_ejes_euros(ax, formatear_x=True, formatear_y=True):
    """Aplica formato euros a ejes de matplotlib (separador de miles + símbolo €)."""
    if formatear_y:
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f} €'))
    if formatear_x:
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f} €'))

# ---------- Flujo principal ----------
def main():
    # 1) Cargar dataset
    try:
        df = cargar_csv(CSV_PATH)
    except Exception as e:
        print("Error cargando CSV:", e)
        sys.exit(1)

    try:
        comprobar_columnas(df)
    except Exception as e:
        print("Error con las columnas del CSV:", e)
        sys.exit(1)

    X = df[X_COLS]
    y = df[Y_COLS]

    # 2) División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3) Cargar o entrenar modelo
    if os.path.exists(MODEL_PATH):
        model = cargar_modelo(MODEL_PATH)
    else:
        model = entrenar_modelo(X_train, y_train)
        guardar_modelo(model, MODEL_PATH)

    # 4) Predicción sobre test
    y_pred = model.predict(X_test)

    # 5) Métricas: R2 y MSE por ciudad (imprimir R2)
    print("\nMétricas por ciudad (test set):")
    metrics = []
    for i, col in enumerate(Y_COLS):
        r2 = r2_score(y_test[col], y_pred[:, i])
        mse = mean_squared_error(y_test[col], y_pred[:, i])
        metrics.append({"ciudad": col, "r2": r2, "mse": mse})
        print(f" - {col}: R2 = {r2:.4f} | MSE = {mse:.2f}")
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_by_city.csv"), index=False)

    # ---------- GRÁFICOS ----------
    # A) Scatter Real vs Predicho por ciudad (formato euros en ambos ejes)
    for i, col in enumerate(Y_COLS):
        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(y_test[col], y_pred[:, i], alpha=0.6)
        mn = min(y_test[col].min(), y_pred[:, i].min())
        mx = max(y_test[col].max(), y_pred[:, i].max())
        ax.plot([mn, mx], [mn, mx], linestyle="--", color="gray")
        ax.set_xlabel("Real (€)")
        ax.set_ylabel("Predicho (€)")
        ax.set_title(f"Real vs Predicho — {col} (R2={metrics[i]['r2']:.3f})")
        formatear_ejes_euros(ax, formatear_x=True, formatear_y=True)
        ax.grid(True)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"real_vs_pred_{col}.png")
        plt.savefig(path, dpi=150)
        plt.show()

    # B) Importancias medias de variables (no es en euros)
    # Promediar importances de cada RandomForest (cada salida)
    importances_matrix = []
    for est in model.estimators_:
        try:
            importances_matrix.append(est.feature_importances_)
        except Exception:
            importances_matrix.append(np.zeros(len(X_COLS)))
    importances_matrix = np.array(importances_matrix)
    mean_importances = importances_matrix.mean(axis=0)
    imp_df = pd.DataFrame({"feature": X_COLS, "importance": mean_importances}).sort_values(by="importance", ascending=False)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "importancias_variables.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(imp_df['feature'], imp_df['importance'])
    ax.set_title("Importancia media de variables")
    ax.set_ylabel("Importancia (unidad adimensional)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    imp_png = os.path.join(OUTPUT_DIR, "importancias_variables.png")
    plt.savefig(imp_png, dpi=150)
    plt.show()

    # C) Comparación por muestras (primeras 6)
    n = min(6, X_test.shape[0])
    sample_X = X_test.iloc[:n].reset_index(drop=True)
    sample_y = y_test.iloc[:n].reset_index(drop=True)
    sample_pred = y_pred[:n, :].astype(int)

    # Guardar CSV comparativo
    comp_df = sample_X.copy()
    for i, col in enumerate(Y_COLS):
        comp_df[f"real_{col}"] = sample_y[col]
        comp_df[f"pred_{col}"] = sample_pred[:, i]
    comp_csv = os.path.join(OUTPUT_DIR, "comparacion_muestras.csv")
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nComparación de muestras guardada en: {comp_csv}")

    # Barras comparativas por muestra (formato euros en eje Y y etiquetas encima)
    labels = [c.replace("presupuesto_", "").capitalize() for c in Y_COLS]
    for idx in range(n):
        reales = sample_y.iloc[idx].values.astype(float)
        preds = sample_pred[idx].astype(float)
        fig, ax = plt.subplots(figsize=(10,4))
        x = np.arange(len(labels))
        ax.bar(x - 0.2, reales, width=0.4, label="Real")
        ax.bar(x + 0.2, preds, width=0.4, label="Predicho")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("Euros")
        ax.set_title(f"Real vs Predicho — muestra {idx+1}")
        formatear_ejes_euros(ax, formatear_x=False, formatear_y=True)
        # Anotar valores encima de cada barra
        for i, v in enumerate(reales):
            ax.text(i - 0.2, v + max(reales.max(), preds.max())*0.01, f"{int(v):,} €", ha="center", va="bottom")
        for i, v in enumerate(preds):
            ax.text(i + 0.2, v + max(reales.max(), preds.max())*0.01, f"{int(v):,} €", ha="center", va="bottom")
        ax.legend()
        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, f"barras_comparacion_muestra_{idx+1}.png")
        plt.savefig(p, dpi=150)
        plt.show()

    # D) Predicción de ejemplo y grafico de barras (formato euros)
    ejemplo = {
        "superficie": 150,
        "habitaciones": 3,
        "precio_m2_madrid": 5700,
        "precio_m2_barcelona": 5000,
        "precio_m2_valencia": 3000,
        "precio_m2_sevilla": 2700,
        "precio_m2_bilbao": 3600,
        "precio_m2_malaga": 3450,
        "precio_m2_jaen": 1300
    }
    X_ej = np.array([[ejemplo["superficie"], ejemplo["habitaciones"],
                      ejemplo["precio_m2_madrid"], ejemplo["precio_m2_barcelona"], ejemplo["precio_m2_valencia"],
                      ejemplo["precio_m2_sevilla"], ejemplo["precio_m2_bilbao"], ejemplo["precio_m2_malaga"],
                      ejemplo["precio_m2_jaen"]]])
    pred_ej = model.predict(X_ej)[0].astype(int)
    labels_short = ["Madrid","Barcelona","Valencia","Sevilla","Bilbao","Málaga","Jaén"]

    fig, ax = plt.subplots(figsize=(10,4))
    x = np.arange(len(labels_short))
    ax.bar(x, pred_ej)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=45)
    ax.set_ylabel("Euros")
    ax.set_title("Predicción presupuesto por ciudad — caso de ejemplo")
    formatear_ejes_euros(ax, formatear_x=False, formatear_y=True)
    for i, v in enumerate(pred_ej):
        ax.text(i, v + max(pred_ej)*0.01, f"{v:,} €", ha="center", va="bottom")
    plt.tight_layout()
    ej_png = os.path.join(OUTPUT_DIR, "prediccion_ejemplo.png")
    plt.savefig(ej_png, dpi=150)
    plt.show()

    print("\nScript finalizado. Todas las figuras también se guardaron en la carpeta 'outputs/'.")

if __name__ == "__main__":
    main()
