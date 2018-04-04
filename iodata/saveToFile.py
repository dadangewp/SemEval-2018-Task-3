'''
description: save files for different purposes (analysis, further steps, ...)
parameter: DATA_DIR (path to file)
'''
import numpy as np
import unicodecsv
import codecs

#variables
DATA_DIR = "D:\\PhD\\RumourEval\\Small Project on Stance Detection in Rumour\\"


def saveTextToFile(text,filename):
  
    myfile = open(DATA_DIR+filename,"w") 
    try:
        myfile.write(text)
    except:
        myfile.write("Not possible to write text")
    myfile.close()
    
    
def saveMatrixToCSVFile(matrix,filename):     
    
    myfile = open(DATA_DIR + filename, "w")
    for l in matrix:
        try:
            for v in l:
                myfile.write(str(v) + ";")
            myfile.write("\n")
        except:
            myfile.write("error")
            
    myfile.close()

def saveTweetToCSVFile(matrix,filename):     
    myfile = codecs.open(DATA_DIR + filename, "w", encoding="utf8")
    for l in matrix:
        try:
            myfile.write(str(l) + "||")
            myfile.write("\n")
        except:
            myfile.write("error")
            myfile.write("\n")
    myfile.close()
    