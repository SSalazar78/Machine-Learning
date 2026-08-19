# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:14:56 2026

@author: Susana A.S.R
"""
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:/Users/Susana A.S.R/Tesseract/tesseract.exe'

img = cv2.imread("Img1.jpg")
text = pytesseract.image_to_string(img, lang = "spa")
print(text)
cv2.imshow("Imagen", img)
cv2.waitKey(0)
cv2.destroyAllWindows()