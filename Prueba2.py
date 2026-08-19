# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 15:05:47 2026

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


# ============================================================
# LIMPIEZA DEL OCR
# ============================================================

df['conf'] = pd.to_numeric(
    df['conf'],
    errors='coerce'
)

# Eliminar texto vacío
df = df[
    (df['text'].notna()) &
    (df['text'].str.strip() != '') &
    (df['conf'] > 0)
].copy()


# ============================================================
# ELIMINAR CARACTERES QUE EL OCR CONFUNDE CON COLUMNAS
# ============================================================

# El OCR está interpretando las líneas verticales de la tabla
# como "|". No queremos que participen en la reconstrucción.

df['text'] = df['text'].astype(str)

df = df[
    ~df['text'].str.fullmatch(r'[|¦]+')
].copy()


# ============================================================
# CALCULAR CENTRO DE CADA PALABRA
# ============================================================

df['centro_x'] = (
    df['left'] + df['width'] / 2
)

df['centro_y'] = (
    df['top'] + df['height'] / 2
)


# ============================================================
# AGRUPAR PALABRAS POR RENGLÓN
# ============================================================

df = df.sort_values(
    ['centro_y', 'centro_x']
).reset_index(drop=True)


