"""Genera un dataset sintético de envíos con el medio de transporte recomendado.

Cada fila es un paquete por enviar. La etiqueta (medio_recomendado) se calcula
con reglas que imitan las limitaciones reales de cada medio: peso, volumen,
distancia, clima, zona, horario, tráfico y urgencia. Después se agrega ruido
para que el problema no sea una tabla de reglas perfecta.

Uso:
    python generar_dataset.py
    python generar_dataset.py --filas 20000 --semilla 7 --salida envios.csv
    python generar_dataset.py --faltantes 0.02   # deja celdas vacías para practicar limpieza
"""

import argparse
import csv
import math
import random

MEDIOS = ("bicicleta", "moto", "camioneta", "dron")

LIMITES = {
    "bicicleta": {"peso_kg": 8, "volumen_l": 40, "distancia_km": 7},
    "moto": {"peso_kg": 25, "volumen_l": 90, "distancia_km": 40},
    "camioneta": {"peso_kg": 1000, "volumen_l": 3000, "distancia_km": 1000},
    "dron": {"peso_kg": 2.5, "volumen_l": 12, "distancia_km": 15},
}

VELOCIDAD_KMH = {"bicicleta": 15, "moto": 35, "camioneta": 30, "dron": 70}
PREPARACION_MIN = {"bicicleta": 5, "moto": 5, "camioneta": 12, "dron": 6}
COSTO_BASE = {"bicicleta": 20, "moto": 35, "camioneta": 90, "dron": 40}
COSTO_KM = {"bicicleta": 2, "moto": 4, "camioneta": 9, "dron": 3}

# El dron vuela en línea recta: recorre menos que la distancia por calle.
FACTOR_LINEA_RECTA_DRON = 0.75

# Cuánto pesa cada minuto de espera según la urgencia del envío.
PESO_TIEMPO = {"normal": 0.3, "express": 1.0, "mismo_dia": 2.0}

ZONAS = (("centro", 0.30), ("residencial", 0.40), ("periferia", 0.20), ("rural", 0.10))
DISTANCIA_POR_ZONA = {
    "centro": (0.3, 6),
    "residencial": (1, 15),
    "periferia": (5, 30),
    "rural": (15, 60),
}
CLIMAS = (("despejado", 0.70), ("lluvia", 0.20), ("viento_fuerte", 0.10))
PRIORIDADES = (("normal", 0.55), ("express", 0.30), ("mismo_dia", 0.15))
HORAS_PICO = {7, 8, 9, 13, 14, 15, 18, 19, 20}

COLUMNAS = [
    "id_envio",
    "distancia_km",
    "peso_kg",
    "largo_cm",
    "ancho_cm",
    "alto_cm",
    "volumen_l",
    "fragil",
    "refrigerado",
    "valor_declarado_mxn",
    "prioridad",
    "zona_destino",
    "clima",
    "hora_salida",
    "trafico",
    "medio_recomendado",
]

# Columnas que pueden quedar vacías con --faltantes. La etiqueta y el id nunca.
COLUMNAS_CON_FALTANTES = ["peso_kg", "volumen_l", "valor_declarado_mxn", "clima", "trafico"]


def elegir(rng, opciones):
    valores, pesos = zip(*opciones)
    return rng.choices(valores, weights=pesos, k=1)[0]


def generar_paquete(rng):
    zona = elegir(rng, ZONAS)
    minimo, maximo = DISTANCIA_POR_ZONA[zona]
    distancia = rng.triangular(minimo, maximo, minimo + (maximo - minimo) * 0.35)

    if rng.random() < 0.06:
        peso = rng.uniform(20, 80)
    else:
        peso = rng.lognormvariate(math.log(2), 1.0)
    peso = min(max(peso, 0.1), 80)

    volumen_objetivo = min(max(peso * rng.uniform(1.5, 8), 0.3), 400)
    lado = (volumen_objetivo * 1000) ** (1 / 3)
    largo = lado * rng.uniform(1.0, 1.8)
    ancho = lado * rng.uniform(0.7, 1.2)
    alto = volumen_objetivo * 1000 / (largo * ancho)
    largo, ancho, alto = round(largo, 1), round(ancho, 1), round(max(alto, 1.0), 1)
    volumen = largo * ancho * alto / 1000

    hora = rng.choices(range(24), weights=[1, 1, 1, 1, 1, 2, 4, 8, 9, 9, 9, 9,
                                           9, 9, 9, 9, 9, 9, 8, 7, 5, 4, 2, 1])[0]
    if hora in HORAS_PICO:
        trafico = elegir(rng, (("bajo", 0.1), ("medio", 0.3), ("alto", 0.6)))
    else:
        trafico = elegir(rng, (("bajo", 0.5), ("medio", 0.35), ("alto", 0.15)))

    return {
        "distancia_km": round(distancia, 2),
        "peso_kg": round(peso, 2),
        "largo_cm": largo,
        "ancho_cm": ancho,
        "alto_cm": alto,
        "volumen_l": round(volumen, 2),
        "fragil": int(rng.random() < 0.20),
        "refrigerado": int(rng.random() < 0.05),
        "valor_declarado_mxn": round(min(max(rng.lognormvariate(math.log(800), 1.2), 50), 80000), 2),
        "prioridad": elegir(rng, PRIORIDADES),
        "zona_destino": zona,
        "clima": elegir(rng, CLIMAS),
        "hora_salida": hora,
        "trafico": trafico,
    }


def es_posible(medio, p):
    limite = LIMITES[medio]
    if p["peso_kg"] > limite["peso_kg"] or p["volumen_l"] > limite["volumen_l"]:
        return False
    if p["distancia_km"] > limite["distancia_km"]:
        return False
    if p["refrigerado"] and medio != "camioneta":
        return False
    noche = p["hora_salida"] >= 21 or p["hora_salida"] < 6
    if medio == "dron":
        if p["clima"] != "despejado" or noche or p["zona_destino"] == "centro":
            return False
    if medio == "bicicleta" and p["zona_destino"] == "rural":
        return False
    return True


def puntaje(medio, p, rng):
    """Costo total estimado: dinero + tiempo ponderado por urgencia + riesgos. Menor es mejor."""
    velocidad = VELOCIDAD_KMH[medio]

    factor_trafico = {
        "bajo": {"bicicleta": 1.0, "moto": 1.0, "camioneta": 1.0, "dron": 1.0},
        "medio": {"bicicleta": 1.0, "moto": 0.9, "camioneta": 0.75, "dron": 1.0},
        "alto": {"bicicleta": 0.95, "moto": 0.8, "camioneta": 0.5, "dron": 1.0},
    }[p["trafico"]][medio]
    velocidad *= factor_trafico

    if p["zona_destino"] == "centro" and medio == "camioneta":
        velocidad *= 0.8
    if p["zona_destino"] == "rural" and medio in ("moto", "camioneta"):
        velocidad *= 1.2

    penalizacion = 0.0
    if p["clima"] == "lluvia":
        if medio == "bicicleta":
            velocidad *= 0.7
            penalizacion += 40
        elif medio == "moto":
            velocidad *= 0.85
            penalizacion += 15
    elif p["clima"] == "viento_fuerte":
        if medio == "bicicleta":
            velocidad *= 0.85
        elif medio == "moto":
            velocidad *= 0.95

    noche = p["hora_salida"] >= 21 or p["hora_salida"] < 6
    if noche and medio == "bicicleta":
        penalizacion += 30

    if p["fragil"]:
        penalizacion += {"bicicleta": 30, "moto": 50, "camioneta": 0, "dron": 20}[medio]

    if p["valor_declarado_mxn"] > 20000:
        penalizacion += {"bicicleta": 25, "moto": 10, "camioneta": 0, "dron": 40}[medio]

    recorrido = p["distancia_km"] * (FACTOR_LINEA_RECTA_DRON if medio == "dron" else 1.0)
    tiempo_min = PREPARACION_MIN[medio] + recorrido / velocidad * 60
    costo = COSTO_BASE[medio] + COSTO_KM[medio] * recorrido
    total = costo + PESO_TIEMPO[p["prioridad"]] * tiempo_min + penalizacion

    # Ruido: dos operadores con la misma información no siempre deciden igual.
    return total * rng.gauss(1.0, 0.08)


def recomendar(p, rng, ruido_etiqueta):
    posibles = [m for m in MEDIOS if es_posible(m, p)]
    mejor = min(posibles, key=lambda m: puntaje(m, p, rng))
    if len(posibles) > 1 and rng.random() < ruido_etiqueta:
        mejor = rng.choice([m for m in posibles if m != mejor])
    return mejor


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filas", type=int, default=10000)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--ruido", type=float, default=0.03,
                        help="proporción de etiquetas cambiadas por otro medio posible (0.03 = 3%%)")
    parser.add_argument("--faltantes", type=float, default=0.0,
                        help="proporción de celdas vacías en algunas columnas (0.02 = 2%%)")
    parser.add_argument("--salida", default="envios_medios.csv")
    args = parser.parse_args()

    rng = random.Random(args.semilla)
    conteo = {m: 0 for m in MEDIOS}

    with open(args.salida, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        escritor.writeheader()
        for i in range(1, args.filas + 1):
            paquete = generar_paquete(rng)
            medio = recomendar(paquete, rng, args.ruido)
            conteo[medio] += 1

            fila = {"id_envio": f"ENV-{i:06d}", **paquete, "medio_recomendado": medio}
            for columna in COLUMNAS_CON_FALTANTES:
                if rng.random() < args.faltantes:
                    fila[columna] = ""
            escritor.writerow(fila)

    print(f"Se escribieron {args.filas} filas en {args.salida}")
    for medio, n in conteo.items():
        print(f"  {medio:<10} {n:>6}  ({n / args.filas:.1%})")


if __name__ == "__main__":
    main()
