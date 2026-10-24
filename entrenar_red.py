"""Entrena una red neuronal (MLP) con el dataset de envíos y la guarda en un archivo.

Requiere: pip install pandas scikit-learn joblib

Uso:
    python entrenar_red.py
    python entrenar_red.py --salida modelo_red.joblib
"""

import argparse

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAS = [
    "distancia_km", "peso_kg", "largo_cm", "ancho_cm", "alto_cm", "volumen_l",
    "fragil", "refrigerado", "valor_declarado_mxn", "hora_salida",
]
CATEGORICAS = ["prioridad", "zona_destino", "clima", "trafico"]
OBJETIVO = "medio_recomendado"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datos", default="envios_medios.csv")
    parser.add_argument("--salida", default="modelo_red.joblib")
    args = parser.parse_args()

    datos = pd.read_csv(args.datos)
    x = datos[NUMERICAS + CATEGORICAS]
    y = datos[OBJETIVO]

    x_entreno, x_prueba, y_entreno, y_prueba = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # La red necesita números en escala parecida; el árbol no.
    preparacion = ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAS),
    ])

    modelo = Pipeline([
        ("preparacion", preparacion),
        ("red", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=800,
            random_state=42,
        )),
    ])
    modelo.fit(x_entreno, y_entreno)
    prediccion = modelo.predict(x_prueba)

    print(classification_report(y_prueba, prediccion, digits=3))
    etiquetas = sorted(y.unique())
    print("Matriz de confusión:")
    print(pd.DataFrame(
        confusion_matrix(y_prueba, prediccion, labels=etiquetas),
        index=[f"real {e}" for e in etiquetas],
        columns=[f"pred {e}" for e in etiquetas],
    ), end="\n\n")

    joblib.dump(modelo, args.salida)
    print(f"Modelo guardado en {args.salida}")


if __name__ == "__main__":
    main()
