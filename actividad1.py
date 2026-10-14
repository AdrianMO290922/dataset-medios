import pandas as pd
from generar_dataset import es_posible, LIMITES
#Se cargan los datos
pd.set_option("display.width", 200)
datos = pd.read_csv("envios_medios.csv")
#Anotamos las predicciones para que queden documentadas
# ¿Qué medio domina con lluvia? ¿y en el centro?
#Busca el valor mas alto de cada fila
print(pd.crosstab(datos["clima"], datos["medio_recomendado"], normalize="index").round(3))
print(pd.crosstab(datos["zona_destino"], datos["medio_recomendado"], normalize="index").round(3))

# ¿Distancia más larga de una bicicleta?
#Compara el max con el LIMIT de la bicicleta
print(datos.groupby("medio_recomendado")["distancia_km"].agg(["min", "median", "max"]))

# ¿Hay etiquetas que rompen las reglas?
#Si false aparece 0 vecees ninguna de las etiquetas esta rota
datos["valida"] = datos.apply(lambda f: es_posible(f["medio_recomendado"], f), axis=1)
print(datos["valida"].value_counts())
