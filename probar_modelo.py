"""Prueba un modelo guardado con un paquete que ustedes describen desde la terminal.

Primero hay que entrenarlo con: python entrenar_arbol.py

Solo la distancia y el peso son obligatorios; lo demás tiene un valor común.

Ejemplos:
    python probar_modelo.py --distancia 5 --peso 2
    python probar_modelo.py --distancia 12 --peso 0.5 --medidas 15 10 8 --hora 3
    python probar_modelo.py --distancia 30 --peso 3 --refrigerado --zona rural
    python probar_modelo.py --distancia 9 --peso 0.5 --modelo modelo_bosque.joblib
    python probar_modelo.py --help
"""

import argparse

import pandas as pd

from usar_modelo import cargar_modelo, es_posible

parser = argparse.ArgumentParser(description="Recomienda el medio de transporte para un paquete.")
parser.add_argument("--distancia", type=float, required=True, help="kilómetros hasta el destino")
parser.add_argument("--peso", type=float, required=True, help="kilogramos")
parser.add_argument("--medidas", type=float, nargs=3, default=[20, 15, 10],
                    metavar=("LARGO", "ANCHO", "ALTO"), help="centímetros (por defecto 20 15 10)")
parser.add_argument("--fragil", action="store_true")
parser.add_argument("--refrigerado", action="store_true")
parser.add_argument("--valor", type=float, default=500, help="valor declarado en pesos")
parser.add_argument("--prioridad", choices=["normal", "express", "mismo_dia"], default="normal")
parser.add_argument("--zona", choices=["centro", "residencial", "periferia", "rural"], default="residencial")
parser.add_argument("--clima", choices=["despejado", "lluvia", "viento_fuerte"], default="despejado")
parser.add_argument("--hora", type=int, choices=range(24), default=12, metavar="0-23")
parser.add_argument("--trafico", choices=["bajo", "medio", "alto"], default="medio")
parser.add_argument("--modelo", default="modelo_medios.joblib", help="archivo del modelo entrenado")
args = parser.parse_args()

largo, ancho, alto = args.medidas
paquete = {
    "distancia_km": args.distancia,
    "peso_kg": args.peso,
    "largo_cm": largo,
    "ancho_cm": ancho,
    "alto_cm": alto,
    "volumen_l": round(largo * ancho * alto / 1000, 2),
    "fragil": int(args.fragil),
    "refrigerado": int(args.refrigerado),
    "valor_declarado_mxn": args.valor,
    "prioridad": args.prioridad,
    "zona_destino": args.zona,
    "clima": args.clima,
    "hora_salida": args.hora,
    "trafico": args.trafico,
}

modelo = cargar_modelo(args.modelo)
probabilidades = modelo.predict_proba(pd.DataFrame([paquete]))[0]
ranking = sorted(zip(modelo.classes_, probabilidades), key=lambda par: -par[1])

print(f"Modelo: {args.modelo}")
print(f"Paquete: {paquete['distancia_km']} km, {paquete['peso_kg']} kg, {paquete['volumen_l']} L, "
      f"{paquete['zona_destino']}, {paquete['clima']}, {paquete['hora_salida']}:00, "
      f"prioridad {paquete['prioridad']}"
      f"{', frágil' if paquete['fragil'] else ''}{', refrigerado' if paquete['refrigerado'] else ''}\n")

print("Lo que opina el modelo:")
for medio, probabilidad in ranking:
    nota = "" if es_posible(medio, paquete) else "   <- no es posible, lo descartan las reglas"
    print(f"  {medio:<10} {probabilidad:6.1%}{nota}")

posibles = [medio for medio, _ in ranking if es_posible(medio, paquete)]
print(f"\nMedio recomendado: {posibles[0] if posibles else 'camioneta'}")
