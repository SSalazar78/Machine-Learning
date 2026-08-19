# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:19:49 2026

@author: Susana A.S.R
"""
import os

# ==============================
# CONFIGURACIÓN DE PADDLE
# ==============================

# Servidor de descarga
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"

# Desactivar oneDNN / MKLDNN
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

# Desactivar PIR
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

# ==============================
# IMPORTACIONES
# ==============================

from paddlex import create_pipeline

# ==============================
# CONFIGURACIÓN
# ==============================

imagen = "Fallo913.jpg"
carpeta_salida = "./resultado_paddle"

# Crear carpeta de salida
os.makedirs(carpeta_salida, exist_ok=True)

# ==============================
# CREAR PIPELINE
# ==============================

pipeline = create_pipeline(
    pipeline="PP-StructureV3"
)

# ==============================
# PROCESAMIENTO
# ==============================

resultado = pipeline.predict(
    input=imagen,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# ==============================
# GUARDAR RESULTADOS
# ==============================

for res in resultado:

    res.print()

    res.save_to_json(
        save_path=carpeta_salida
    )

    res.save_to_html(
        save_path=carpeta_salida
    )

    res.save_to_xlsx(
        save_path=carpeta_salida
    )

    res.save_to_img(
        save_path=carpeta_salida
    )

print("PROCESAMIENTO TERMINADO")
print("Resultados guardados en:")
print(os.path.abspath(carpeta_salida))
