'''
description: implementations of different classifier (using scikit) and settings
'''

from sklearn import svm
from sklearn import tree
from sklearn import gaussian_process
from sklearn import ensemble
from sklearn import metrics
from sklearn import naive_bayes
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neural_network import MLPClassifier
import numpy as np
from iodata.saveToFile import saveTweetToCSVFile


def SVMclassifierTrain(featureMatrix,labelMatrix,featureTest,labelTest):
    
    clf = svm.SVC()
    #selector = RFE(clf, 11, step=1)
    #selector = selector.fit(featureMatrix, labelMatrix)
    print (clf.fit(featureMatrix, labelMatrix))  
    #print ("Ukuran FeatureTest :"+str(len(labelMatrix)))
    # cross validation
    #print(cross_val_score(clf, featureMatrix, labelMatrix, cv=10, scoring="accuracy"))  
    #predicted = cross_val_predict(clf,featureMatrix,labelMatrix,cv=10) 
    #predict = clf.predict(featureMatrix)
    #print(len(predict))
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=10, scoring="f1")
    print("F1 Score (Cross-V): %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #score = metrics.accuracy_score(labelMatrix, predicted)
    #print(cross_val_score(clf, X_train, Y_train.iloc[:,i], cv=3, scoring='neg_log_loss'))
    #print ("F1 Score = "+ str(score))
    #TASK = "A" # Define, A or B
    #FNAME = './predictions-task' + TASK + '.txt'
    #PREDICTIONSFILE = open(FNAME, "w")
    #for p in predicted:
    #    PREDICTIONSFILE.write("{}\n".format(p))
    #PREDICTIONSFILE.close()
    #print(classification_report_imbalanced(labelTest, pipeline.predict(labelMatrix)))
    #labelTest = np.asarray(labelTest)
    #misclassified = clf.predict(featureTest)
    #saveTweetToCSVFile(misclassified,"featureMatrixTweet.txt")
    #print (misclassified);
    #print (clf.decision_function)
    return clf


def classifierPredict(featureMatrix,model):
    return model.predict(featureMatrix) 


def DecisionTreeTrain(featureMatrix,labelMatrix):
    
    clf = tree.DecisionTreeClassifier()
    print (clf.fit(featureMatrix, labelMatrix))  
    
    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=10)  
    #predicted = cross_val_predict(clf,featureMatrix,labelMatrix,cv=10)    
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #score = metrics.f1_score(labelMatrix, predicted, pos_label=1)
    #print ("F1 Score = "+ str(score))
    return clf

def NaiveBayesTrain(featureMatrix,labelMatrix):
    
    clf = naive_bayes.GaussianNB()
    print (clf.fit(featureMatrix, labelMatrix))  
    
    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=10)      
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    return clf


def GaussianProcessTrain(featureMatrix,labelMatrix):
    
    clf = gaussian_process.GaussianProcessClassifier()  
    #print (clf.fit(featureMatrix, labelMatrix))  
    
    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=10)      
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    return clf


def AdaBoostTrain(featureMatrix,labelMatrix):
    
    clf = ensemble.AdaBoostClassifier()  
    #print (clf.fit(featureMatrix, labelMatrix))  
    
    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=10)      
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    return clf


def RandomForestTrain(featureMatrix,labelMatrix):
    
    clf = ensemble.RandomForestClassifier(criterion="entropy")  
    print (clf.fit(featureMatrix, labelMatrix))  
    
    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=10)  
    #predicted = cross_val_predict(clf,featureMatrix,labelMatrix,cv=10)    
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #score = metrics.f1_score(labelMatrix, predicted, pos_label=1)
    #print ("F1 Score = "+ str(score))
    return clf


def NeuroNetTrain(featureMatrix,labelMatrix):
    
    clf = MLPClassifier(solver='adam',max_iter=400,hidden_layer_sizes=(25,2))  
    print (clf.fit(featureMatrix, labelMatrix))  
    
    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=10)      
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    return clf
