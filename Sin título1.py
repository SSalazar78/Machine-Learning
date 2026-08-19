# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:06:14 2026

@author: Susana A.S.R
"""
import paddle
import paddlex

print("PaddlePaddle:", paddle.__version__)
print("PaddleX:", paddlex.__version__)
print("Dispositivo:", paddle.device.get_device())
