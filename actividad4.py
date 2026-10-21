import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

NUMERICAS = ["distancia_km", "peso_kg", "largo_cm", "ancho_cm", "alto_cm", "volumen_l",
             "fragil", "refrigerado", "valor_declarado_mxn", "hora_salida"]
CATEGORICAS = ["prioridad", "zona_destino", "clima", "trafico"]

datos = pd.read_csv("envios_medios.csv")
x = datos[NUMERICAS + CATEGORICAS]
y = datos["medio_recomendado"]
x_entreno, x_prueba, y_entreno, y_prueba = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

modelo = joblib.load("modelo_medios.joblib")

#Para obtener las columnas mas importantes

nombres = [n.split("__", 1)[1] for n in modelo.named_steps["preparacion"].get_feature_names_out()]
importancias = pd.Series(modelo.named_steps["arbol"].feature_importances_, index=nombres)
print(importancias.sort_values(ascending=False).round(3).to_string())

print(export_text(modelo.named_steps["arbol"], feature_names=nombres, max_depth=2))

# Agrupada
agrupada = {}
for nombre, valor in importancias.items():
    columna = next((c for c in CATEGORICAS if nombre.startswith(c + "_")), nombre)
    agrupada[columna] = agrupada.get(columna, 0) + valor
print(pd.Series(agrupada).sort_values(ascending=False).round(3))
