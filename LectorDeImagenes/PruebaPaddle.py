# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:38:03 2026

@author: Susana A.S.R
"""
import os
from paddlex import create_pipeline
from img2table.document import Image

# Forzar el uso de un servidor de descarga alternativo y más estable (Baidu Object Storage)
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"

# 2. DESACTIVAR oneDNN/MKLDNN (Soluciona el error de onednn_instruction.cc)
os.environ["FLAGS_use_mkldnn"] = "0"

# 3. DESACTIVAR el nuevo ejecutor PIR que está causando el conflicto
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

#Configuracion
imagen = 'tabla_recortada.jpg'
    
carpeta_salida = './resultado_paddle'

#Crear pipeline
pipeline = create_pipeline(pipeline = 'PP-StructureV3')

#Procesamiento de imagen
resultado = pipeline.predict( input = imagen, use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

#Guardar resutaldos
for res in resultado:

    res.print()

    #Diferentes tipos de salida
    res.save_to_html(save_path=carpeta_salida)

    res.save_to_xlsx(save_path=carpeta_salida)


print("PROCESAMIENTO TERMINADO")
print("Resultados guardados en:")
print(carpeta_salida)
print(os.path.abspath(carpeta_salida))