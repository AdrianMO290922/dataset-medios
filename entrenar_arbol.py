"""Entrena un árbol de decisión que recomienda el medio de transporte y lo guarda en un archivo.

Requiere: pip install pandas scikit-learn joblib

Uso:
    python entrenar_arbol.py
    python entrenar_arbol.py --profundidad 6 --salida modelo_medios.joblib
"""

import argparse

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

NUMERICAS = [
    "distancia_km", "peso_kg", "largo_cm", "ancho_cm", "alto_cm", "volumen_l",
    "fragil", "refrigerado", "valor_declarado_mxn", "hora_salida",
]
CATEGORICAS = ["prioridad", "zona_destino", "clima", "trafico"]
OBJETIVO = "medio_recomendado"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datos", default="envios_medios.csv")
    parser.add_argument("--profundidad", type=int, default=8)
    parser.add_argument("--salida", default="modelo_medios.joblib")
    args = parser.parse_args()

    # 1. Cargar los datos y separar lo que el modelo ve (x) de lo que debe adivinar (y).
    datos = pd.read_csv(args.datos)
    x = datos[NUMERICAS + CATEGORICAS]
    y = datos[OBJETIVO]
    print(f"Envíos: {len(datos)}")

    # 2. Apartar el 20 % para evaluar con envíos que el modelo nunca vio.
    x_entreno, x_prueba, y_entreno, y_prueba = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Preparar las columnas: el árbol solo entiende números, así que cada
    #    categoría se convierte en columnas de 0 y 1 (one-hot).
    preparacion = ColumnTransformer([
        ("numericas", "passthrough", NUMERICAS),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])

    # 4. Unir la preparación y el árbol en un solo objeto. Así, al usar el modelo,
    #    basta con darle los datos tal como vienen en el CSV.
    modelo = Pipeline([
        ("preparacion", preparacion),
        ("arbol", DecisionTreeClassifier(
            max_depth=args.profundidad, min_samples_leaf=5, random_state=42
        )),
    ])

    # 5. Entrenar.
    modelo.fit(x_entreno, y_entreno)

    # 6. Evaluar con los envíos apartados.
    prediccion = modelo.predict(x_prueba)
    print(f"Exactitud con datos de prueba: {(prediccion == y_prueba).mean():.1%}\n")
    print(classification_report(y_prueba, prediccion, digits=3))

    etiquetas = sorted(y.unique())
    print("Matriz de confusión (filas: real, columnas: predicción):")
    print(pd.DataFrame(
        confusion_matrix(y_prueba, prediccion, labels=etiquetas),
        index=etiquetas, columns=etiquetas,
    ), end="\n\n")

    # 7. Ver las primeras reglas que aprendió el árbol. Solo se puede con un
    #    árbol: si lo cambian por un Random Forest, este paso se salta.
    if isinstance(modelo.named_steps["arbol"], DecisionTreeClassifier):
        nombres = list(modelo.named_steps["preparacion"].get_feature_names_out())
        nombres = [n.split("__", 1)[1] for n in nombres]
        print("Primeros niveles del árbol:")
        print(export_text(modelo.named_steps["arbol"], feature_names=nombres, max_depth=3))

    # 8. Guardar el modelo entrenado en un archivo. Cualquier programa puede
    #    cargarlo después con joblib.load, sin volver a entrenar.
    joblib.dump(modelo, args.salida)
    print(f"Modelo guardado en {args.salida}")


if __name__ == "__main__":
    main()
