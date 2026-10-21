import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from generar_dataset import es_posible

pd.set_option("display.width", 200)

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
prediccion = modelo.predict(x_prueba)

tabla = x_prueba.assign(real=y_prueba, predicho=prediccion)
errores = tabla[tabla["real"] != tabla["predicho"]]
print("errores:", len(errores), "de", len(tabla))
print(errores.groupby(["real", "predicho"]).size().sort_values(ascending=False))

## Segunda pregunta para saber los drones 
md = errores[(errores["real"] == "moto") & (errores["predicho"] == "dron")].copy()
md["noche"] = (md["hora_salida"] >= 21) | (md["hora_salida"] < 6)
md["dron_posible"] = md.apply(lambda f: es_posible("dron", f), axis=1)

print("total:", len(md))
print("de noche:", md["noche"].sum())
print("dron sí era posible:", md["dron_posible"].sum())
print(md[md["dron_posible"]])        # los que las reglas NO corregirían

#Preguta3 sobre las moto real que el modelo mando en bicilceta
mb = errores[(errores["real"] == "moto") & (errores["predicho"] == "bicicleta")]
print(len(mb))
print(mb["distancia_km"].describe())     # ¿están cerca de 7 km?

acertadas = tabla[(tabla["real"] == "bicicleta") & (tabla["predicho"] == "bicicleta")]

cols = ["distancia_km", "peso_kg", "volumen_l", "fragil", "valor_declarado_mxn", "hora_salida"]
print(pd.DataFrame({"error_moto→bici": mb[cols].median(),
                    "bicis_acertadas": acertadas[cols].median()}))

for col in ["clima", "prioridad", "zona_destino", "trafico"]:
    print(pd.DataFrame({"error": mb[col].value_counts(normalize=True),
                        "acertadas": acertadas[col].value_counts(normalize=True)}).round(2))
