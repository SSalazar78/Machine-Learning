# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:54:19 2026

@author: Susana A.S.R
"""

import pytesseract
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

pytesseract.pytesseract.tesseract_cmd = r'C:/Users/Susana A.S.R/Tesseract/tesseract.exe'

img = cv2.imread("Fallo913.jpg")

#Escala de grises
imgGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(imgGris, cmap= 'gray')
plt.axis('off')
plt.show()

#Aumentar resolucion
imgGrande = cv2.resize(imgGris, None,fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

#OCR
custom_config = r'--oem 3 --psm 6'

datos = pytesseract.image_to_data(
    imgGrande,
    lang='spa',
    config=custom_config,
    output_type=pytesseract.Output.DICT
)

#Crear dataframe con pandas
df = pd.DataFrame(datos)

#Convertir confianza a numero
df['conf'] = pd.to_numeric(df['conf'], errors='coerce')

#Eliminar palabras vacias
df = df[
    (df['text'].notna()) &
    (df['text'].str.strip() != '') &
    (df['conf'] > 0)
].copy()

#Calcular centro de cada palabra
df['centro_x'] = (
    df['left'] +
    df['width'] / 2
)

df['centro_y'] = (
    df['top'] +
    df['height'] / 2
)

#Ordenar las palabras
df = df.sort_values(
    ['centro_y', 'centro_x']
).reset_index(drop=True)

#Agrupar palabras en renglones
filas = []

tolerancia_y = 15

for _, palabra in df.iterrows():

    agregada = False

    for fila in filas:

        # Comparar con el centro vertical promedio
        diferencia = abs(
            palabra['centro_y'] -
            fila['centro_y']
        )

        if diferencia <= tolerancia_y:

            fila['palabras'].append(palabra)

            # Actualizar centro Y
            fila['centro_y'] = np.mean(
                [p['centro_y'] for p in fila['palabras']]
            )

            agregada = True
            break

    if not agregada:

        filas.append({
            'centro_y': palabra['centro_y'],
            'palabras': [palabra]
        })


# Ordenar filas verticalmente
filas = sorted(
    filas,
    key=lambda x: x['centro_y']
)


# ============================================================
# 9. ORDENAR PALABRAS DENTRO DE CADA FILA

for fila in filas:

    fila['palabras'] = sorted(
        fila['palabras'],
        key=lambda x: x['centro_x']
    )


# ============================================================
# 10. MOSTRAR LO QUE TESSERACT ENCONTRÓ
print("\n========== RENGLONES DETECTADOS ==========\n")

for i, fila in enumerate(filas):

    texto = ' '.join(
        palabra['text']
        for palabra in fila['palabras']
    )

    print(
        f"Fila {i}: {texto}"
    )


# ============================================================
# 11. GUARDAR OCR ORGANIZADO POR RENGLÓN

resultado_lineas = []

for fila in filas:

    texto = ' '.join(
        palabra['text']
        for palabra in fila['palabras']
    )

    resultado_lineas.append(
        texto
    )


df_lineas = pd.DataFrame({
    'texto': resultado_lineas
})


df_lineas.to_csv(
    "resultado_lineas.csv",
    index=False,
    encoding='utf-8-sig'
)


print(
    "\nSe guardó: resultado_lineas.csv"
)



