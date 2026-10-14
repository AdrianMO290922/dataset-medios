"""Carga un modelo ya entrenado desde su archivo y recomienda el medio de paquetes nuevos.

Primero hay que entrenarlo con: python entrenar_arbol.py

Uso:
    python usar_modelo.py
    python usar_modelo.py --modelo modelo_bosque.joblib
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


def cargar_modelo(ruta):
    if not Path(ruta).exists():
        sys.exit(f"No encuentro {ruta}. Primero entrenen el modelo con: python entrenar_arbol.py")
    return joblib.load(ruta)


# Límites que ningún modelo puede saltarse. El modelo aprende de ejemplos y
# a veces se equivoca; estas reglas vienen del negocio y no se negocian.
LIMITES = {
    "bicicleta": {"peso_kg": 8, "volumen_l": 40, "distancia_km": 7},
    "moto": {"peso_kg": 25, "volumen_l": 90, "distancia_km": 40},
    "camioneta": {"peso_kg": 1000, "volumen_l": 3000, "distancia_km": 1000},
    "dron": {"peso_kg": 2.5, "volumen_l": 12, "distancia_km": 15},
}


def es_posible(medio, paquete):
    limite = LIMITES[medio]
    if paquete["peso_kg"] > limite["peso_kg"]:
        return False
    if paquete["volumen_l"] > limite["volumen_l"]:
        return False
    if paquete["distancia_km"] > limite["distancia_km"]:
        return False
    if paquete["refrigerado"] and medio != "camioneta":
        return False
    noche = paquete["hora_salida"] >= 21 or paquete["hora_salida"] < 6
    if medio == "dron" and (paquete["clima"] != "despejado" or noche
                            or paquete["zona_destino"] == "centro"):
        return False
    if medio == "bicicleta" and paquete["zona_destino"] == "rural":
        return False
    return True


def recomendar_medio(paquete):
    """Recibe un paquete como diccionario y devuelve el medio, la confianza y las alternativas."""
    datos = pd.DataFrame([paquete])
    probabilidades = modelo.predict_proba(datos)[0]
    ranking = sorted(zip(modelo.classes_, probabilidades), key=lambda par: -par[1])

    for medio, confianza in ranking:
        if es_posible(medio, paquete):
            return {
                "medio": medio,
                "confianza": round(float(confianza), 3),
                "corregido": medio != ranking[0][0],
                "ranking": [(m, round(float(p), 3)) for m, p in ranking],
            }
    return {"medio": "camioneta", "confianza": 0.0, "corregido": True,
            "ranking": [(m, round(float(p), 3)) for m, p in ranking]}


EJEMPLOS = {
    "Libro a 2 km en el centro, hora pico": {
        "distancia_km": 1.99, "peso_kg": 0.99, "largo_cm": 17.8, "ancho_cm": 20.5, "alto_cm": 13.7,
        "volumen_l": 5.0, "fragil": 0, "refrigerado": 0, "valor_declarado_mxn": 210,
        "prioridad": "normal", "zona_destino": "centro", "clima": "despejado",
        "hora_salida": 18, "trafico": "alto",
    },
    "Medicamento ligero a 9 km, periferia, tarde despejada": {
        "distancia_km": 8.95, "peso_kg": 0.54, "largo_cm": 17.3, "ancho_cm": 10.7, "alto_cm": 15.0,
        "volumen_l": 2.78, "fragil": 0, "refrigerado": 0, "valor_declarado_mxn": 789,
        "prioridad": "normal", "zona_destino": "periferia", "clima": "despejado",
        "hora_salida": 16, "trafico": "medio",
    },
    "Paquete ligero a 13 km, a las 3:00 de la madrugada": {
        "distancia_km": 12.64, "peso_kg": 0.31, "largo_cm": 11.7, "ancho_cm": 7.6, "alto_cm": 10.6,
        "volumen_l": 0.94, "fragil": 0, "refrigerado": 0, "valor_declarado_mxn": 4603,
        "prioridad": "mismo_dia", "zona_destino": "periferia", "clima": "despejado",
        "hora_salida": 3, "trafico": "medio",
    },
    "Microondas a 18 km con lluvia": {
        "distancia_km": 18.0, "peso_kg": 14.0, "largo_cm": 55, "ancho_cm": 40, "alto_cm": 35,
        "volumen_l": 77.0, "fragil": 1, "refrigerado": 0, "valor_declarado_mxn": 2500,
        "prioridad": "express", "zona_destino": "periferia", "clima": "lluvia",
        "hora_salida": 17, "trafico": "medio",
    },
    "Vacunas refrigeradas a 30 km": {
        "distancia_km": 30.0, "peso_kg": 3.0, "largo_cm": 30, "ancho_cm": 25, "alto_cm": 25,
        "volumen_l": 18.75, "fragil": 1, "refrigerado": 1, "valor_declarado_mxn": 15000,
        "prioridad": "mismo_dia", "zona_destino": "rural", "clima": "despejado",
        "hora_salida": 8, "trafico": "bajo",
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="modelo_medios.joblib")
    args = parser.parse_args()

    modelo = cargar_modelo(args.modelo)
    print(f"Modelo cargado de {args.modelo}\n")

    for nombre, paquete in EJEMPLOS.items():
        resultado = recomendar_medio(paquete)
        print(nombre)
        print(f"  Medio recomendado: {resultado['medio']} (confianza {resultado['confianza']:.0%})")
        if resultado["corregido"]:
            print(f"  Nota: el modelo prefería {resultado['ranking'][0][0]}, pero no es posible para este paquete.")
        print(f"  Ranking del modelo: {resultado['ranking']}\n")
