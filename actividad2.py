import pandas as pd

datos = pd.read_csv("envios_medios.csv")
pd.set_option("display.max_rows", 50)
#Nos ayuda a saber las horas, la disponibilidad segun el medio y la cantidad de envíos que se hacen en cada hora segun el dataset
tabla = pd.crosstab(datos["hora_salida"], datos["medio_recomendado"])
print(tabla)

tabla.plot(kind="bar", stacked=True, figsize=(12, 6)).figure.savefig("horas.png")
