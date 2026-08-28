# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:38:03 2026

@author: Susana A.S.R
"""
#Para no sobrecargar el nucleo, solo introducir PDF de 25 páginas como máximo
import os
import gc
import fitz

# IMPORTANTE: estas variables deben definirse ANTES de importar paddle o paddlex
# 2. DESACTIVAR oneDNN/MKLDNN (Soluciona el error de onednn_instruction.cc)
# 3. DESACTIVAR el nuevo ejecutor PIR que está causando el conflicto

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

from paddlex import create_pipeline

# Forzar el uso de un servidor de descarga alternativo y más estable (Baidu Object Storage)
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"


#Configuracion. Agregar el nombre del documento
documento = 'PruebaFalloT-21-1-65-1-35.pdf'
    
carpeta_salida = './resultado_paddle'

#Crear pipeline
pipeline = create_pipeline(pipeline = 'PP-StructureV3')

#Abrir el PDF
pdf = fitz.open(documento)

total_de_pagina = len(pdf)

#Procesar una pagina a la vez para evitar errores
for numero_pagina in range(total_de_pagina):
    
    pagina = pdf[numero_pagina]
    
    #Convertir pagina del PDF a imagen
    pix = pagina.get_pixmap(matrix = fitz.Matrix(1.5,1.5), alpha=False)
    
    #Nombre temporal de la imagen
    imagen = os.path.join(carpeta_salida, f"pagina_{numero_pagina + 1}.png")
    
    pix.save(imagen)
    
    #Liberar pixmap
    del pix 
    
    #Procesamiento solo de esa imagen
    try:
        resultado = pipeline.predict( input = documento, use_doc_orientation_classify=False,
                                     use_doc_unwarping=False, use_textline_orientation=False)

        #Guardar resutaldos
        for res in resultado:
    
            #Va a analizar una pagina por cada iteracion i
            res.print()

            #Diferentes tipos de salida
            res.save_to_html(save_path=carpeta_salida)

            res.save_to_xlsx(save_path=carpeta_salida)
            
    except Exception() as e:
        print(f"Error en pagina {numero_pagina + 1}")
        print(e)
        
    #Liberar memoria
    del resultado
    gc.collect()
    
pdf.close()
print("PROCESAMIENTO TERMINADO")
print("Resultados guardados en:")
print(os.path.abspath(carpeta_salida))