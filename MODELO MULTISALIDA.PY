import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# ----------------------------------------------------
# 1. Cargar dataset CSV
# ----------------------------------------------------
df = pd.read_csv(
    r"C:\Users\fredd\Desktop\CURSO ATU #2\DATA SET.csv",
    encoding="latin1"
)

# Variables de entrada (X)
X_cols = [
    "superficie_m2", "habitaciones",
    "precio_m2_madrid", "precio_m2_barcelona", "precio_m2_valencia",
    "precio_m2_sevilla", "precio_m2_bilbao", "precio_m2_malaga", "precio_m2_jaen"
]

# Variables objetivo (los presupuestos por ciudad)
y_cols = [
    "presupuesto_madrid", "presupuesto_barcelona", "presupuesto_valencia",
    "presupuesto_sevilla", "presupuesto_bilbao", "presupuesto_malaga",
    "presupuesto_jaen"
]

X = df[X_cols]
y = df[y_cols]

# ----------------------------------------------------
# 2. División entrenamiento / prueba
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# 3. Crear modelo MULTISALIDA
# ----------------------------------------------------
model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=250, random_state=42)
)

# Entrenar
model.fit(X_train, y_train)

# ----------------------------------------------------
# 4. Evaluación del modelo
# ----------------------------------------------------
y_pred = model.predict(X_test)

print("\nRESULTADOS DEL MODELO MULTISALIDA:\n")
for i, col in enumerate(y_cols):
    r2 = r2_score(y_test[col], y_pred[:, i])
    mse = mean_squared_error(y_test[col], y_pred[:, i])
    print(f"{col}: R2 = {r2:.4f} | MSE = {mse:.2f}")

# ----------------------------------------------------
# 5. Guardar modelo entrenado
# ----------------------------------------------------
with open("modelo_multisalida.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModelo guardado como 'modelo_multisalida.pkl'")

# ----------------------------------------------------
# 6. FUNCIÓN PARA HACER PREDICCIONES NUEVAS
# ----------------------------------------------------
def predecir(superficie, habitaciones,
             precio_m2_madrid, precio_m2_barcelona, precio_m2_valencia,
             precio_m2_sevilla, precio_m2_bilbao, precio_m2_malaga, precio_m2_jaen):

    entrada = np.array([[superficie, habitaciones,
                         precio_m2_madrid, precio_m2_barcelona, precio_m2_valencia,
                         precio_m2_sevilla, precio_m2_bilbao, precio_m2_malaga, precio_m2_jaen]])

    pred = model.predict(entrada)

    print("\nPRESUPUESTO ESTIMADO POR CIUDAD:")
    for i, ciudad in enumerate(y_cols):
        print(f"{ciudad}: {int(pred[0][i])} €")

# ----------------------------------------------------
# 7. EJEMPLO DE USO (MODIFÍCALO SI QUIERES)
# ----------------------------------------------------
print("\nEJEMPLO DE PREDICCIÓN:")
predecir(
    superficie=150,
    habitaciones=3,
    precio_m2_madrid=5700,
    precio_m2_barcelona=5000,
    precio_m2_valencia=3000,
    precio_m2_sevilla=2700,
    precio_m2_bilbao=3600,
    precio_m2_malaga=3450,
    precio_m2_jaen=1300
)
