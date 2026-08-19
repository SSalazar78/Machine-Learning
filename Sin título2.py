# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:27:43 2026

@author: Susana A.S.R
"""
import os
import cv2
import pandas as pd

from img2table.document import Image
from paddleocr import PaddleOCR

# CONFIGURACIÓN

os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"


imagen = "tabla_recortada.jpg"

carpeta_salida = "./resultado_final"

os.makedirs(carpeta_salida, exist_ok=True)


# 1. CARGAR IMAGEN
img = cv2.imread(imagen)

alto, ancho = img.shape[:2]

# 2. DETECTAR LA TABLA CON IMG2TABLE
print("\nDetectando la cuadrícula con img2table...")

documento = Image(src=imagen)

tablas = documento.extract_tables(
    borderless_tables=False,
    implicit_rows=True,
    implicit_columns=True
)

if len(tablas) == 0:
    raise RuntimeError(
        "img2table no encontró ninguna tabla."
    )

print(f"Tablas encontradas: {len(tablas)}")

tabla = tablas[0]

print(f"Filas detectadas: {len(tabla.df)}")
print(f"Columnas detectadas: {len(tabla.df.columns)}")

# 3. INICIAR PADDLEOCR
ocr = PaddleOCR(lang="es")

# 4. FUNCIÓN PARA OCR DE UNA CELDA

def reconocer_celda(recorte):

    if recorte is None:
        return ""

    if recorte.size == 0:
        return ""

    # Agrandar la celda

    escala = 2

    recorte = cv2.resize(
        recorte,
        None,
        fx=escala,
        fy=escala,
        interpolation=cv2.INTER_CUBIC
    )

    # OCR
    resultado = ocr.predict(recorte)

    textos = []

    for res in resultado:

        try:

            data = res.json

            if callable(data):
                data = data()

            if isinstance(data, dict):

                contenido = data.get("res", data)

                textos_ocr = contenido.get(
                    "rec_texts",
                    []
                )

                for texto in textos_ocr:

                    if texto:
                        textos.append(str(texto))

        except Exception as e:

            print(
                "Advertencia al leer una celda:",
                e
            )

    return " ".join(textos).strip()

# ============================================================
# 5. OBTENER LAS CELDAS DE IMG2TABLE
# ============================================================

print("\nObteniendo coordenadas de las celdas...")

# img2table guarda las celdas en el atributo 'content'
# pero no necesariamente como una lista de filas.
# Convertimos el contenido en una lista de celdas.

celdas = tabla.content

print("Tipo de tabla.content:", type(celdas))

# Si content es un diccionario, tomamos sus valores
if isinstance(celdas, dict):
    celdas = list(celdas.values())

# Si es una lista/tupla, la convertimos a lista
else:
    celdas = list(celdas)


print("Número de elementos encontrados:", len(celdas))


# ============================================================
# 6. INSPECCIONAR LAS CELDAS
# ============================================================

print("\nPrimeros elementos encontrados:")

for i, elemento in enumerate(celdas[:5]):

    print(
        i,
        type(elemento),
        elemento
    )


# ============================================================
# 7. EXTRAER CELDAS INDIVIDUALES
# ============================================================

# En algunas versiones de img2table las celdas pueden
# encontrarse agrupadas. Por eso las aplanamos.

celdas_finales = []


def aplanar(objeto):

    if isinstance(objeto, (list, tuple)):

        for elemento in objeto:
            aplanar(elemento)

    elif isinstance(objeto, dict):

        for elemento in objeto.values():
            aplanar(elemento)

    else:

        # Solo guardamos objetos que tengan bbox
        if hasattr(objeto, "bbox"):
            celdas_finales.append(objeto)


aplanar(celdas)


print(
    "\nCeldas con coordenadas encontradas:",
    len(celdas_finales)
)


# ============================================================
# 8. ORDENAR LAS CELDAS
# ============================================================

# Obtener coordenadas

def obtener_bbox(celda):

    bbox = celda.bbox

    return (
        bbox.y1,
        bbox.x1
    )


celdas_finales.sort(
    key=obtener_bbox
)


# ============================================================
# 9. OCR CON PADDLE
# ============================================================

print("\nComenzando OCR...")


def reconocer_celda(recorte):

    if recorte is None or recorte.size == 0:
        return ""

    # Aumentar tamaño
    recorte = cv2.resize(
        recorte,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    resultado = ocr.predict(recorte)

    textos = []

    for res in resultado:

        try:

            data = res.json

            if callable(data):
                data = data()

            if isinstance(data, dict):

                datos_res = data.get(
                    "res",
                    data
                )

                textos_ocr = datos_res.get(
                    "rec_texts",
                    []
                )

                textos.extend(
                    str(t)
                    for t in textos_ocr
                    if t
                )

        except Exception as e:

            print(
                "Error OCR:",
                e
            )

    return " ".join(textos).strip()


# ============================================================
# 10. RECONSTRUIR FILAS Y COLUMNAS
# ============================================================

# Primero obtenemos bbox y texto

resultados_celdas = []


for celda in celdas_finales:

    bbox = celda.bbox

    x1 = max(
        0,
        int(bbox.x1)
    )

    y1 = max(
        0,
        int(bbox.y1)
    )

    x2 = min(
        ancho,
        int(bbox.x2)
    )

    y2 = min(
        alto,
        int(bbox.y2)
    )

    recorte = img[
        y1:y2,
        x1:x2
    ]

    texto = reconocer_celda(
        recorte
    )

    resultados_celdas.append(
        {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "texto": texto
        }
    )


# ============================================================
# 11. AGRUPAR POR FILAS
# ============================================================

# Las celdas que tienen una coordenada Y similar
# pertenecen a la misma fila.

resultados_celdas.sort(
    key=lambda c: (
        c["y1"],
        c["x1"]
    )
)


filas = []

tolerancia_y = 10


for celda in resultados_celdas:

    agregada = False

    for fila in filas:

        y_promedio = sum(
            c["y1"]
            for c in fila
        ) / len(fila)

        if abs(
            celda["y1"] - y_promedio
        ) <= tolerancia_y:

            fila.append(celda)

            agregada = True

            break

    if not agregada:

        filas.append(
            [celda]
        )


# Ordenar cada fila de izquierda a derecha

for fila in filas:

    fila.sort(
        key=lambda c: c["x1"]
    )


# ============================================================
# 12. CREAR DATAFRAME
# ============================================================

datos = [
    [celda["texto"] for celda in fila]
    for fila in filas
]


df = pd.DataFrame(datos)


# ============================================================
# 13. LIMPIAR FILAS VACÍAS
# ============================================================

df = df.replace(
    r"^\s*$",
    pd.NA,
    regex=True
)

df = df.dropna(
    axis=0,
    how="all"
)

df = df.fillna("")


# ============================================================
# 14. GUARDAR CSV
# ============================================================

ruta_csv = os.path.join(
    carpeta_salida,
    "tabla_final.csv"
)

df.to_csv(
    ruta_csv,
    index=False,
    header=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. GUARDAR EXCEL
# ============================================================

ruta_excel = os.path.join(
    carpeta_salida,
    "tabla_final.xlsx"
)

df.to_excel(
    ruta_excel,
    index=False,
    header=False
)


# ============================================================
# 16. RESULTADO
# ============================================================

print("\n========================================")
print("PROCESAMIENTO TERMINADO")
print("========================================")

print("\nFilas:", len(df))
print("Columnas:", len(df.columns))

print("\nResultado:")
print(
    df.to_string(
        index=False,
        header=False
    )
)

print("\nCSV:")
print(
    os.path.abspath(ruta_csv)
)

print("\nExcel:")
print(
    os.path.abspath(ruta_excel)
)