import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from generar_dataset import es_posible

pd.set_option("display.width", 250)

NUMERICAS = ["distancia_km", "peso_kg", "largo_cm", "ancho_cm", "alto_cm", "volumen_l",
             "fragil", "refrigerado", "valor_declarado_mxn", "hora_salida"]
CATEGORICAS = ["prioridad", "zona_destino", "clima", "trafico"]
COLS = NUMERICAS + CATEGORICAS

datos = pd.read_csv("envios_medios.csv")
x = datos[COLS]
y = datos["medio_recomendado"]
x_entreno, x_prueba, y_entreno, y_prueba = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)


def evaluar(nombre, xe, ye, class_weight=None, xp=x_prueba, yp=y_prueba):
    """Entrena un árbol igual al del paso 6 y lo mide siempre con la prueba (xp, yp)."""
    preparacion = ColumnTransformer([
        ("numericas", "passthrough", NUMERICAS),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])
    m = Pipeline([
        ("preparacion", preparacion),
        ("arbol", DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                                         random_state=42, class_weight=class_weight)),
    ])
    m.fit(xe, ye)
    pred = m.predict(xp)
    imposibles = sum(not es_posible(p, fila) for p, (_, fila) in zip(pred, xp.iterrows()))
    return {
        "estrategia": nombre,
        "envios_entreno": len(ye),
        "exactitud": round((pred == yp).mean(), 3),
        "precision_dron": round(precision_score(yp, pred, labels=["dron"], average=None, zero_division=0)[0], 3),
        "recall_dron": round(recall_score(yp, pred, labels=["dron"], average=None, zero_division=0)[0], 3),
        "veces_predice_dron": int((pred == "dron").sum()),
        "sugerencias_imposibles": imposibles,
    }

filas = []
entreno = x_entreno.assign(medio=y_entreno)          # entrenamiento con la etiqueta
n_dron = (entreno["medio"] == "dron").sum()
n_mayor = entreno["medio"].value_counts().max()

# 0. Sin balancear
filas.append(evaluar("sin balancear", x_entreno, y_entreno))

# 1. Quitar ejemplos de las clases que sobran (todas bajan al tamaño del dron)
sub = pd.concat([g.sample(n_dron, random_state=42) for _, g in entreno.groupby("medio")])
filas.append(evaluar("quitar (undersampling)", sub[COLS], sub["medio"]))

# 2. Repetir ejemplos de las clases que faltan (todas suben al tamaño de la moto)
rep = pd.concat([entreno] + [
    g.sample(n_mayor - len(g), replace=True, random_state=42)
    for _, g in entreno.groupby("medio") if len(g) < n_mayor
])
filas.append(evaluar("repetir (oversampling)", rep[COLS], rep["medio"]))

# 3. Ejemplos nuevos de dron (del dataset generado con otra semilla)
extra = pd.read_csv("envios_extra.csv")
nuevos = extra[extra["medio_recomendado"] == "dron"]
xe = pd.concat([x_entreno, nuevos[COLS]])
ye = pd.concat([y_entreno, nuevos["medio_recomendado"]])
filas.append(evaluar("drones nuevos", xe, ye))

# 4. Decirle al modelo que fallar un dron cuesta más
filas.append(evaluar("class_weight=balanced", x_entreno, y_entreno, class_weight="balanced"))


filas = []
entreno = x_entreno.assign(medio=y_entreno)          # entrenamiento con la etiqueta
n_dron = (entreno["medio"] == "dron").sum()
n_mayor = entreno["medio"].value_counts().max()

# 0. Sin balancear
filas.append(evaluar("sin balancear", x_entreno, y_entreno))

# 1. Quitar ejemplos de las clases que sobran (todas bajan al tamaño del dron)
sub = pd.concat([g.sample(n_dron, random_state=42) for _, g in entreno.groupby("medio")])
filas.append(evaluar("quitar (undersampling)", sub[COLS], sub["medio"]))

# 2. Repetir ejemplos de las clases que faltan (todas suben al tamaño de la moto)
rep = pd.concat([entreno] + [
    g.sample(n_mayor - len(g), replace=True, random_state=42)
    for _, g in entreno.groupby("medio") if len(g) < n_mayor
])
filas.append(evaluar("repetir (oversampling)", rep[COLS], rep["medio"]))

# 3. Ejemplos nuevos de dron (del dataset generado con otra semilla)
extra = pd.read_csv("envios_extra.csv")
nuevos = extra[extra["medio_recomendado"] == "dron"]
xe = pd.concat([x_entreno, nuevos[COLS]])
ye = pd.concat([y_entreno, nuevos["medio_recomendado"]])
filas.append(evaluar("drones nuevos", xe, ye))

# 4. Decirle al modelo que fallar un dron cuesta más
filas.append(evaluar("class_weight=balanced", x_entreno, y_entreno, class_weight="balanced"))

# Trampa A: repetir drones ANTES de separar (los duplicados caen en entrenamiento y prueba)
drones = datos[datos["medio_recomendado"] == "dron"]
todo = pd.concat([datos, drones.sample(3600, replace=True, random_state=42)])
xa, xb, ya, yb = train_test_split(
    todo[COLS], todo["medio_recomendado"],
    test_size=0.2, stratify=todo["medio_recomendado"], random_state=42,
)
filas.append(evaluar("TRAMPA: repetir antes de separar", xa, ya, xp=xb, yp=yb))

# Trampa B: balancear también la prueba
prueba = x_prueba.assign(medio=y_prueba)
n_min = prueba["medio"].value_counts().min()
prueba_bal = pd.concat([g.sample(n_min, random_state=42) for _, g in prueba.groupby("medio")])
filas.append(evaluar("TRAMPA: prueba balanceada", sub[COLS], sub["medio"],
                     xp=prueba_bal[COLS], yp=prueba_bal["medio"]))

print(pd.DataFrame(filas).to_string(index=False))
print("drones reales en la prueba:", (y_prueba == "dron").sum())
