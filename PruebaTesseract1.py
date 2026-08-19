# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:39:57 2026

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


#PREPROCESAMIENTO DE LA IMAGEN
#Pasamos a escala de grises
imgGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Fallo913.jpg', imgGris)
#cv2.waitKey(0)  # Espera a que presiones una tecla
#cv2.destroyAllWindows()  # Cierra la ventana

plt.imshow(imgGris, cmap='gray')
plt.axis('off')
plt.show()

#APLICANCO OCR 
#Aumentar la resolución porque las letras son pequeñas
#Se multiplica por 2 o 3 el tamaño
imgGrande = cv2.resize(imgGris, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)


#Optimizacion de la tabla
custom_config = r'--oem 3 --psm 6'

#Uso de Tesseract
datos = pytesseract.image_to_data(imgGrande, lang='spa', config=custom_config, output_type=pytesseract.Output.DICT)
df = pd.DataFrame(datos)


#Filtrar solo palabras con confianza > 0 y texto no vacío
#palabras = df[df['level'] == 5].copy()
#palabras = palabras[(palabras['conf'] > 0) & (palabras['text'].str.strip() != '')]


#Agrupar por líneas nativas de Tesseract
#lineas = palabras.groupby(['block_num', 'line_num']).apply(lambda x: ' '.join(x['text'])).reset_index(name='texto')

#Mostrar resultados
#print("TEXTO EXTRAÍDO")
#for idx, row in lineas.iterrows():
#    print(f"{row['texto']}")

# 9. Guardar en archivo de texto
#with open("resultado_ocr.txt", "w", encoding="utf-8") as f:
#    for idx, row in lineas.iterrows():
#        f.write(f"{row['texto']}\n")


# 10. Guardar también en CSV para análisis
#lineas.to_csv("resultado_ocr.csv", index=False, encoding='utf-8')
#print(f"Guardado en 'resultado_ocr.csv'")