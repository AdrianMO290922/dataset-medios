"""Entrena un Random Forest con el dataset de envíos, muestra qué tan bien predice y lo guarda en un archivo.

Requiere: pip install pandas scikit-learn joblib

Uso:
    python entrenar_modelo.py
    python entrenar_modelo.py --datos envios_medios.csv --salida modelo_bosque.joblib
"""

import argparse

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERICAS = [
    "distancia_km", "peso_kg", "largo_cm", "ancho_cm", "alto_cm", "volumen_l",
    "fragil", "refrigerado", "valor_declarado_mxn", "hora_salida",
]
CATEGORICAS = ["prioridad", "zona_destino", "clima", "trafico"]
OBJETIVO = "medio_recomendado"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datos", default="envios_medios.csv")
    parser.add_argument("--salida", default="modelo_bosque.joblib")
    args = parser.parse_args()

    datos = pd.read_csv(args.datos)
    x = datos[NUMERICAS + CATEGORICAS]
    y = datos[OBJETIVO]

    x_entreno, x_prueba, y_entreno, y_prueba = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    preparacion = ColumnTransformer([
        ("numericas", SimpleImputer(strategy="median"), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAS),
    ])

    modelo = Pipeline([
        ("preparacion", preparacion),
        ("bosque", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    modelo.fit(x_entreno, y_entreno)
    prediccion = modelo.predict(x_prueba)

    print(classification_report(y_prueba, prediccion, digits=3))

    etiquetas = sorted(y.unique())
    matriz = pd.DataFrame(
        confusion_matrix(y_prueba, prediccion, labels=etiquetas),
        index=[f"real {e}" for e in etiquetas],
        columns=[f"pred {e}" for e in etiquetas],
    )
    print("Matriz de confusión:")
    print(matriz, end="\n\n")

    nombres = modelo.named_steps["preparacion"].get_feature_names_out()
    importancias = pd.Series(
        modelo.named_steps["bosque"].feature_importances_, index=nombres
    ).sort_values(ascending=False)
    print("Variables más importantes:")
    print(importancias.head(10).round(3).to_string(), end="\n\n")

    ejemplo = pd.DataFrame([{
        "distancia_km": 6.5, "peso_kg": 1.2, "largo_cm": 25, "ancho_cm": 18, "alto_cm": 10,
        "volumen_l": 4.5, "fragil": 0, "refrigerado": 0, "valor_declarado_mxn": 900,
        "prioridad": "express", "zona_destino": "residencial", "clima": "despejado",
        "hora_salida": 11, "trafico": "alto",
    }])
    probabilidades = modelo.predict_proba(ejemplo)[0]
    print("Paquete de ejemplo: 6.5 km, 1.2 kg, express, residencial, despejado, tráfico alto")
    for medio, prob in sorted(zip(modelo.classes_, probabilidades), key=lambda t: -t[1]):
        print(f"  {medio:<10} {prob:.1%}")

    # El bosque son 300 árboles, así que el archivo pesa decenas de megas.
    joblib.dump(modelo, args.salida)
    print(f"\nModelo guardado en {args.salida}")


if __name__ == "__main__":
    main()
