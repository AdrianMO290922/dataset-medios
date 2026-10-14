# Dataset sintético: ¿qué medio de transporte conviene para este paquete?

Cada fila del archivo `envios_medios.csv` es un paquete que la empresa de
entregas tiene que enviar, junto con el medio de transporte recomendado:
`bicicleta`, `moto`, `camioneta` o `dron`. Sirve para entrenar un modelo de
clasificación que, dado un paquete nuevo, sugiera el medio más adecuado.

Los datos son **sintéticos**: no vienen de una empresa real. Se generaron con
reglas que imitan las limitaciones de cada medio, y después se les agregó
ruido para que se parezcan a decisiones humanas, que no siempre son
perfectas.

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `envios_medios.csv` | El dataset listo para usar: 10 000 envíos. |
| `generar_dataset.py` | El generador. Solo usa la biblioteca estándar de Python. |
| `entrenar_arbol.py` | Entrena el árbol de decisión, muestra sus reglas y lo guarda en `modelo_medios.joblib`. |
| `usar_modelo.py` | Carga un modelo guardado y recomienda el medio para paquetes nuevos, validando las reglas del negocio. Primero hay que correr `entrenar_arbol.py`. Con `--modelo` se elige qué archivo cargar. |
| `probar_modelo.py` | Prueba un modelo guardado con un paquete que se describe desde la terminal, por ejemplo `python probar_modelo.py --distancia 9 --peso 0.5 --hora 3`. `--help` muestra todas las opciones. |
| `entrenar_modelo.py` | Un Random Forest para comparar resultados. Lo guarda en `modelo_bosque.joblib` (unos 50 MB). |

Los modelos entrenados quedan guardados como archivos `.joblib` en la carpeta
desde donde se ejecutó el script. Se pueden copiar y reutilizar en cualquier
programa con `joblib.load`, sin volver a entrenar.

La explicación paso a paso está en los apuntes, en la sección
*La IA dentro de la plataforma: un modelo que recomienda el medio de transporte*.

## Diccionario de datos

| Columna | Tipo | Descripción |
|---|---|---|
| `id_envio` | texto | Identificador del envío (`ENV-000001`). No se usa para entrenar. |
| `distancia_km` | número | Distancia por calle desde el almacén hasta el destino. |
| `peso_kg` | número | Peso del paquete, de 0.1 a 80 kg. |
| `largo_cm`, `ancho_cm`, `alto_cm` | número | Medidas de la caja. |
| `volumen_l` | número | Volumen en litros. Se calcula con las tres medidas. |
| `fragil` | 0 / 1 | 1 si el contenido es frágil. |
| `refrigerado` | 0 / 1 | 1 si necesita cadena de frío. |
| `valor_declarado_mxn` | número | Valor del contenido en pesos. |
| `prioridad` | categoría | `normal`, `express` o `mismo_dia`. |
| `zona_destino` | categoría | `centro`, `residencial`, `periferia` o `rural`. |
| `clima` | categoría | `despejado`, `lluvia` o `viento_fuerte`. |
| `hora_salida` | entero | Hora del día en que sale el paquete, de 0 a 23. |
| `trafico` | categoría | `bajo`, `medio` o `alto`. Es más probable que sea alto en horas pico. |
| `medio_recomendado` | categoría | **Etiqueta a predecir**: `bicicleta`, `moto`, `camioneta` o `dron`. |

Así quedan repartidas las etiquetas con la configuración por defecto:

| Medio | Envíos | Porcentaje |
|---|---|---|
| moto | 4578 | 45.8 % |
| bicicleta | 2796 | 28.0 % |
| camioneta | 1678 | 16.8 % |
| dron | 948 | 9.5 % |

El dron es la clase minoritaria a propósito: en la realidad solo se puede usar
en pocas condiciones.

## Cómo se decidió la etiqueta

El generador calcula la etiqueta en dos pasos.

**1. Qué medios son posibles.** Un medio se descarta si el paquete no cabe o
si las condiciones no lo permiten:

| Medio | Peso máximo | Volumen máximo | Distancia máxima | Otras restricciones |
|---|---|---|---|---|
| bicicleta | 8 kg | 40 L | 7 km | No va a zonas rurales. |
| moto | 25 kg | 90 L | 40 km | — |
| camioneta | sin límite práctico | sin límite práctico | sin límite práctico | Es el único medio para paquetes refrigerados. |
| dron | 2.5 kg | 12 L | 15 km | Solo con clima despejado, de 6:00 a 20:59 y fuera del centro (espacio aéreo restringido). |

**2. Cuál de los posibles conviene más.** A cada medio posible se le calcula un
costo total, y gana el más bajo:

- **Dinero:** una tarifa base más un costo por kilómetro. La camioneta es la
  más cara y la bicicleta la más barata.
- **Tiempo:** depende de la velocidad de cada medio. El tráfico frena mucho a
  la camioneta, un poco a la moto y casi nada a la bicicleta. Al dron no lo
  frena, y además vuela en línea recta. La lluvia y el viento frenan a la
  bicicleta y a la moto.
- **Urgencia:** en un envío `mismo_dia`, cada minuto pesa casi siete veces más
  que en uno `normal`, así que ganan los medios rápidos aunque sean más caros.
- **Riesgos:** los paquetes frágiles penalizan a la moto, a la bicicleta y al
  dron. Los de valor alto penalizan al dron y a la bicicleta. La bicicleta de
  noche y con lluvia también recibe penalizaciones.

Después se agrega ruido de dos formas. El costo de cada medio varía un 8 %
al azar, y un 3 % de las etiquetas se cambian por otro medio que también
era posible. Por eso ningún modelo va a llegar al 100 %, y eso está bien: los
datos reales tampoco son perfectos.

## Uso

Generar el dataset (no hace falta instalar nada):

```bash
python generar_dataset.py
```

Opciones:

```bash
python generar_dataset.py --filas 20000            # otra cantidad de envíos (por defecto, 10 000)
python generar_dataset.py --semilla 7              # otro dataset distinto, también reproducible
python generar_dataset.py --ruido 0.10             # etiquetas más ruidosas (10 %)
python generar_dataset.py --faltantes 0.02         # deja un 2 % de celdas vacías para practicar limpieza
python generar_dataset.py --salida mis_envios.csv
```

Con la misma semilla siempre sale exactamente el mismo archivo.

Entrenar el árbol de decisión y probarlo:

```bash
pip install pandas scikit-learn joblib
python entrenar_arbol.py
python usar_modelo.py
```

Entrenar el Random Forest para comparar y usarlo:

```bash
python entrenar_modelo.py
python usar_modelo.py --modelo modelo_bosque.joblib
```

Resultado con el dataset por defecto (20 % de los datos para prueba):

```
              precision    recall  f1-score   support

   bicicleta      0.906     0.934     0.920       559
   camioneta      0.997     0.863     0.925       336
        dron      0.914     0.953     0.933       190
        moto      0.930     0.951     0.941       915

    accuracy                          0.931      2000
```

Si su modelo queda muy por debajo de 93 %, algo se puede mejorar. Si queda muy
por encima, revisen que no estén evaluando con los mismos datos con los que
entrenaron.

## Ideas para trabajar con el dataset

- Comparar un árbol de decisión pequeño con el Random Forest. El árbol se
  puede dibujar: ¿se parecen sus reglas a las de la tabla de arriba?
- Quitar `volumen_l` y ver si el modelo lo reconstruye a partir de las tres
  medidas.
- Generar el dataset con `--faltantes 0.05` y decidir cómo llenar los huecos.
- Fijarse en el dron: es la clase con menos ejemplos. ¿Qué pasa con su
  `recall` si se entrena con `class_weight="balanced"`?
- Revisar los errores de la matriz de confusión: ¿en qué paquetes se
  confunden la bicicleta y la moto? ¿Tiene sentido?

## Cómo se conecta con la plataforma

En la plataforma de entregas, este modelo haría el papel del recomendador:
recibe los datos de un paquete y devuelve un medio. El modelo no reemplaza
a los patrones del curso. Su respuesta llega a Django a través de un Adapter,
como el de `route_hint`, y el medio que sugiere se convierte en una de las
estrategias de entrega. Si mañana se cambia el modelo por otro mejor, el resto
de la plataforma no debería enterarse.
