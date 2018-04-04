# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 01:57:50 2017

@author: dadangewp
"""
from matplotlib import pyplot as plt

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()