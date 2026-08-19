# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:16:27 2026

@author: Susana A.S.R
"""

from img2table.document import Image

documento = Image(src="tabla_recortada.jpg")

tablas = documento.extract_tables(
    borderless_tables=False,
    implicit_rows=True,
    implicit_columns=True
)

tabla = tablas[0]

print(type(tabla))
print(tabla)

print(type(tabla.content))
print(tabla.content)