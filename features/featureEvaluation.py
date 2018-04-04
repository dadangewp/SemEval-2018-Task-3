'''
description: feature evaluations/analysis
'''
from sklearn.feature_selection import chi2, mutual_info_classif

def featureClassCoerr(featureMatix,labelMatrix): 
    
    c, p = chi2(featureMatix, labelMatrix)
    
    #print ("------------------------- feature chi 2 begin (chi2) -----------------")
    #print (c)
    #print ("------------------------- feature chi 2 (p-val) ----------------------")
    #print (p)
    #print ("------------------------- feature chi 2 end --------------------------")
    
    m = mutual_info_classif(featureMatix, labelMatrix)
    #print ("------------------------- feature mutual_info_classif 2 begin---------")
    #print (m)
    #print ("------------------------- feature mutual_info_classif 2 end---------")