# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 11:41:21 2020

@author: timot
"""
from pycocotools.coco import COCO

coco = COCO('../lemon-dataset/annotations/instances_default.json')
print(coco)