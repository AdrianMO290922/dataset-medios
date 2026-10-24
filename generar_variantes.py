"""Genera 5 variantes del dataset de envíos para practicar balanceo y clases nuevas.

1. envios_desbalanceado.csv   — casi solo moto; el dron es muy raro
2. envios_balanceado.csv      — las 4 clases casi a la par
3. envios_leve_desbalance.csv — desbalance suave (parecido al original, un poco más marcado)
4. envios_muy_desbalanceado.csv — moto domina; bicicleta y dron casi desaparecen
5. envios_con_eskuter.csv     — misma lógica + clase nueva «eskuter»

Uso:
    python generar_variantes.py
    python generar_variantes.py --filas 10000
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path

import generar_dataset as g

SALIDA = Path(__file__).resolve().parent


def es_posible_con_eskuter(medio, p):
    if medio == "eskuter":
        # Entre bici y moto: ligero, volumen chico, no rural, no noche, no lluvia fuerte.
        if p["peso_kg"] > 12 or p["volumen_l"] > 50 or p["distancia_km"] > 20:
            return False
        if p["refrigerado"]:
            return False
        if p["zona_destino"] == "rural":
            return False
        noche = p["hora_salida"] >= 21 or p["hora_salida"] < 6
        if noche or p["clima"] == "viento_fuerte":
            return False
        return True
    return g.es_posible(medio, p)


def puntaje_con_eskuter(medio, p, rng):
    if medio == "eskuter":
        velocidad = 28.0
        if p["trafico"] == "medio":
            velocidad *= 0.92
        elif p["trafico"] == "alto":
            velocidad *= 0.85
        if p["clima"] == "lluvia":
            velocidad *= 0.8
            penalizacion = 20.0
        else:
            penalizacion = 0.0
        if p["fragil"]:
            penalizacion += 35
        if p["valor_declarado_mxn"] > 20000:
            penalizacion += 15
        tiempo_min = 4 + p["distancia_km"] / velocidad * 60
        # Un poco más caro que la bici/moto: aparece, pero no domina.
        costo = 33 + 3.8 * p["distancia_km"]
        total = costo + g.PESO_TIEMPO[p["prioridad"]] * tiempo_min + penalizacion
        return total * rng.gauss(1.0, 0.08)
    return g.puntaje(medio, p, rng)


def recomendar(medios, p, rng, ruido, es_posible, puntaje):
    posibles = [m for m in medios if es_posible(m, p)]
    if not posibles:
        return "camioneta"
    mejor = min(posibles, key=lambda m: puntaje(m, p, rng))
    if len(posibles) > 1 and rng.random() < ruido:
        mejor = rng.choice([m for m in posibles if m != mejor])
    return mejor


def sesgar_paquete(rng, sesgo: str):
    """Ajusta el generador para empujar ciertas clases arriba o abajo."""
    p = g.generar_paquete(rng)

    if sesgo == "desbalanceado":
        # Muchas distancias medias/largas y clima malo → moto/camioneta.
        if rng.random() < 0.55:
            p["distancia_km"] = round(rng.uniform(8, 35), 2)
            p["zona_destino"] = rng.choice(["periferia", "rural", "residencial"])
        if rng.random() < 0.45:
            p["clima"] = rng.choice(["lluvia", "viento_fuerte"])
        if rng.random() < 0.25:
            p["hora_salida"] = rng.choice([0, 1, 2, 3, 4, 5, 22, 23])
        if rng.random() < 0.15:
            p["peso_kg"] = round(rng.uniform(9, 22), 2)

    elif sesgo == "balanceado":
        # Cuotas aproximadas forzando perfiles “típicos” de cada medio.
        perfil = rng.choice(["bicicleta", "moto", "camioneta", "dron"])
        if perfil == "bicicleta":
            p["distancia_km"] = round(rng.uniform(0.5, 5.5), 2)
            p["peso_kg"] = round(rng.uniform(0.2, 5), 2)
            p["volumen_l"] = round(rng.uniform(0.5, 25), 2)
            p["zona_destino"] = rng.choice(["centro", "residencial"])
            p["clima"] = "despejado"
            p["refrigerado"] = 0
            p["hora_salida"] = rng.randint(8, 18)
        elif perfil == "dron":
            p["distancia_km"] = round(rng.uniform(3, 12), 2)
            p["peso_kg"] = round(rng.uniform(0.1, 2.0), 2)
            p["volumen_l"] = round(rng.uniform(0.3, 10), 2)
            p["zona_destino"] = rng.choice(["residencial", "periferia"])
            p["clima"] = "despejado"
            p["refrigerado"] = 0
            p["hora_salida"] = rng.randint(8, 18)
            p["prioridad"] = rng.choice(["express", "mismo_dia", "normal"])
        elif perfil == "camioneta":
            if rng.random() < 0.5:
                p["refrigerado"] = 1
                p["peso_kg"] = round(rng.uniform(1, 40), 2)
            else:
                p["peso_kg"] = round(rng.uniform(30, 70), 2)
                p["volumen_l"] = round(rng.uniform(100, 350), 2)
            p["distancia_km"] = round(rng.uniform(10, 50), 2)
            p["zona_destino"] = rng.choice(["periferia", "rural", "residencial"])
        else:  # moto
            p["distancia_km"] = round(rng.uniform(6, 25), 2)
            p["peso_kg"] = round(rng.uniform(1, 18), 2)
            p["volumen_l"] = round(rng.uniform(5, 70), 2)
            p["zona_destino"] = rng.choice(["residencial", "periferia", "centro"])
            if rng.random() < 0.35:
                p["clima"] = "lluvia"

    elif sesgo == "leve":
        # Un poco más de moto y menos de dron que el original.
        if rng.random() < 0.25:
            p["distancia_km"] = round(rng.uniform(8, 30), 2)
        if rng.random() < 0.18:
            p["clima"] = "lluvia"

    elif sesgo == "muy_desbalanceado":
        if rng.random() < 0.7:
            p["distancia_km"] = round(rng.uniform(10, 38), 2)
            p["peso_kg"] = round(rng.uniform(5, 22), 2)
            p["zona_destino"] = rng.choice(["periferia", "rural"])
        if rng.random() < 0.55:
            p["clima"] = rng.choice(["lluvia", "viento_fuerte", "lluvia"])
        if rng.random() < 0.35:
            p["hora_salida"] = rng.choice([0, 1, 2, 3, 4, 5, 21, 22, 23])
        if rng.random() < 0.2:
            p["refrigerado"] = 1

    elif sesgo == "eskuter":
        # De vez en cuando un perfil donde el eskuter puede ganar (clase nueva, minoría).
        if rng.random() < 0.12:
            p["distancia_km"] = round(rng.uniform(2, 15), 2)
            p["peso_kg"] = round(rng.uniform(0.5, 10), 2)
            p["volumen_l"] = round(rng.uniform(2, 40), 2)
            p["zona_destino"] = rng.choice(["centro", "residencial", "periferia"])
            p["clima"] = rng.choice(["despejado", "despejado", "lluvia"])
            p["refrigerado"] = 0
            p["hora_salida"] = rng.randint(8, 19)
            p["prioridad"] = rng.choice(["normal", "express"])

    # Recalcular medidas coherentes si se movió volumen/peso a mano.
    if p["volumen_l"] > 0 and (sesgo in {"balanceado", "eskuter"}):
        lado = (p["volumen_l"] * 1000) ** (1 / 3)
        p["largo_cm"] = round(lado * rng.uniform(1.0, 1.5), 1)
        p["ancho_cm"] = round(lado * rng.uniform(0.8, 1.2), 1)
        p["alto_cm"] = round(max(p["volumen_l"] * 1000 / (p["largo_cm"] * p["ancho_cm"]), 1.0), 1)
        p["volumen_l"] = round(p["largo_cm"] * p["ancho_cm"] * p["alto_cm"] / 1000, 2)

    return p


def una_fila(rng, medios, sesgo, ruido, es_posible, puntaje, forzar_medio=None):
    """Genera un paquete. Si forzar_medio, reintenta hasta que la etiqueta coincida."""
    for _ in range(800):
        paquete = sesgar_paquete(rng, sesgo if forzar_medio is None else "balanceado")
        if forzar_medio is not None:
            # Empujar el perfil hacia el medio pedido.
            paquete = sesgar_paquete(rng, "balanceado")
            # Reusar el bloque de perfil del medio forzado:
            perfil_rng = random.Random(rng.random())
            # regenerar con el perfil exacto:
            p = g.generar_paquete(rng)
            if forzar_medio == "bicicleta":
                p["distancia_km"] = round(rng.uniform(0.5, 5.5), 2)
                p["peso_kg"] = round(rng.uniform(0.2, 5), 2)
                p["volumen_l"] = round(rng.uniform(0.5, 25), 2)
                p["zona_destino"] = rng.choice(["centro", "residencial"])
                p["clima"] = "despejado"
                p["refrigerado"] = 0
                p["hora_salida"] = rng.randint(8, 18)
            elif forzar_medio == "dron":
                p["distancia_km"] = round(rng.uniform(3, 12), 2)
                p["peso_kg"] = round(rng.uniform(0.1, 2.0), 2)
                p["volumen_l"] = round(rng.uniform(0.3, 10), 2)
                p["zona_destino"] = rng.choice(["residencial", "periferia"])
                p["clima"] = "despejado"
                p["refrigerado"] = 0
                p["hora_salida"] = rng.randint(8, 18)
            elif forzar_medio == "camioneta":
                if rng.random() < 0.55:
                    p["refrigerado"] = 1
                    p["peso_kg"] = round(rng.uniform(1, 40), 2)
                    p["volumen_l"] = round(rng.uniform(5, 80), 2)
                else:
                    p["peso_kg"] = round(rng.uniform(30, 70), 2)
                    p["volumen_l"] = round(rng.uniform(100, 350), 2)
                p["distancia_km"] = round(rng.uniform(10, 50), 2)
                p["zona_destino"] = rng.choice(["periferia", "rural", "residencial"])
            elif forzar_medio == "eskuter":
                p["distancia_km"] = round(rng.uniform(2, 14), 2)
                p["peso_kg"] = round(rng.uniform(0.5, 10), 2)
                p["volumen_l"] = round(rng.uniform(2, 40), 2)
                p["zona_destino"] = rng.choice(["centro", "residencial", "periferia"])
                p["clima"] = "despejado"
                p["refrigerado"] = 0
                p["hora_salida"] = rng.randint(8, 19)
                p["prioridad"] = rng.choice(["normal", "express"])
            else:  # moto
                p["distancia_km"] = round(rng.uniform(6, 25), 2)
                p["peso_kg"] = round(rng.uniform(1, 18), 2)
                p["volumen_l"] = round(rng.uniform(5, 70), 2)
                p["zona_destino"] = rng.choice(["residencial", "periferia", "centro"])
                if rng.random() < 0.4:
                    p["clima"] = "lluvia"
            lado = (p["volumen_l"] * 1000) ** (1 / 3)
            p["largo_cm"] = round(lado * rng.uniform(1.0, 1.5), 1)
            p["ancho_cm"] = round(lado * rng.uniform(0.8, 1.2), 1)
            p["alto_cm"] = round(max(p["volumen_l"] * 1000 / (p["largo_cm"] * p["ancho_cm"]), 1.0), 1)
            p["volumen_l"] = round(p["largo_cm"] * p["ancho_cm"] * p["alto_cm"] / 1000, 2)
            paquete = p
            del perfil_rng

        medio = recomendar(medios, paquete, rng, ruido, es_posible, puntaje)
        if forzar_medio is None or medio == forzar_medio:
            return paquete, medio
    # Si no se logró forzar, devolver lo último.
    return paquete, medio


def escribir(ruta, filas, medios, sesgo, semilla, ruido, con_eskuter=False, forzar_balance=False):
    rng = random.Random(semilla)
    es_posible = es_posible_con_eskuter if con_eskuter else g.es_posible
    puntaje = puntaje_con_eskuter if con_eskuter else g.puntaje
    conteo = Counter()
    filas_datos = []

    if forzar_balance:
        base = filas // len(medios)
        resto = filas - base * len(medios)
        cupos = {m: base for m in medios}
        for m in list(medios)[:resto]:
            cupos[m] += 1
        for medio_obj, cupo in cupos.items():
            for _ in range(cupo):
                paquete, medio = una_fila(
                    rng, medios, sesgo, ruido, es_posible, puntaje, forzar_medio=medio_obj
                )
                filas_datos.append((paquete, medio))
        rng.shuffle(filas_datos)
    else:
        for _ in range(filas):
            paquete, medio = una_fila(rng, medios, sesgo, ruido, es_posible, puntaje)
            filas_datos.append((paquete, medio))

    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=g.COLUMNAS)
        escritor.writeheader()
        for i, (paquete, medio) in enumerate(filas_datos, start=1):
            conteo[medio] += 1
            escritor.writerow({
                "id_envio": f"ENV-{i:06d}",
                **paquete,
                "medio_recomendado": medio,
            })

    print(f"\n{ruta.name}  ({filas} filas, semilla {semilla})")
    for medio in medios:
        n = conteo[medio]
        print(f"  {medio:<10} {n:>6}  ({n / filas:.1%})")
    return conteo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filas", type=int, default=10000)
    args = parser.parse_args()

    variantes = [
        # nombre, sesgo, semilla, medios, con_eskuter, forzar_balance
        ("envios_desbalanceado.csv", "desbalanceado", 101,
         ("bicicleta", "moto", "camioneta", "dron"), False, False),
        ("envios_balanceado.csv", "balanceado", 202,
         ("bicicleta", "moto", "camioneta", "dron"), False, True),
        ("envios_leve_desbalance.csv", "leve", 303,
         ("bicicleta", "moto", "camioneta", "dron"), False, False),
        ("envios_muy_desbalanceado.csv", "muy_desbalanceado", 404,
         ("bicicleta", "moto", "camioneta", "dron"), False, False),
        ("envios_con_eskuter.csv", "eskuter", 505,
         ("bicicleta", "moto", "camioneta", "dron", "eskuter"), True, False),
    ]

    for nombre, sesgo, semilla, medios, con_eskuter, forzar_balance in variantes:
        escribir(
            SALIDA / nombre,
            args.filas,
            medios,
            sesgo,
            semilla,
            ruido=0.03,
            con_eskuter=con_eskuter,
            forzar_balance=forzar_balance,
        )

    print("\nListos en", SALIDA)


if __name__ == "__main__":
    main()
