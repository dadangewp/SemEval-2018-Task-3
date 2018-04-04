# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:53:07 2017

@author: dadangewp
"""
from features.preprocessingIronyDetection import featureExtraction
from learning.classification import SVMclassifierTrain, classifierPredict, DecisionTreeTrain, GaussianProcessTrain, AdaBoostTrain, RandomForestTrain, NeuroNetTrain, NaiveBayesTrain
from sklearn.model_selection import train_test_split
import sklearn.metrics as scikitm
from iodata.readIronyData import parse_training, parse_testing
import gensim

"Using SVM F1-Score = 0.6759"
USE_TEST = False
CLASSIFIER = "SVM"
DIR = "D:\\PhD\\RumourEval\\Small Project on Irony Detection\\SemEval2018-T3-train-taskA.txt"
DIR_TEST = "D:\\PhD\\RumourEval\\Small Project on Irony Detection\\SemEval2018-Task3-master\\SemEval2018-Task3-master\\datasets\\test_TaskA\\SemEval2018-T3_input_test_taskA_annotated.txt"
#DIR_TEST = "D:\\PhD\\RumourEval\\Small Project on Irony Detection\\SemEval2018-Task3-master-gold\\SemEval2018-Task3-master\\datasets\\goldtest_TaskB\\SemEval2018-T3_gold_test_taskB_emoji.txt"
if __name__ == '__main__':
    
    #model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\dadangewp\\Stance Detection in Rumor on Social Media\\GoogleNews-vectors-negative300.bin', binary=True)
    print ("started ...")
    model = "zonk"
    # read Training data
    dataTrain, dataLabel= parse_training(DIR)
    print ("Training data read") 
    #print (len(dataTrain))
    featureMatrixTrain, labelMatrixTrain, tweetMatrixTrain, feature_names = featureExtraction(dataTrain, dataLabel, model)
    #saveMatrixToCSVFile(featureMatrixTrain,"Feature Matrix Train.csv")
    #saveMatrixToCSVFile(tweetMatrixTrain,"Tweet Matrix Train.csv")
    #saveMatrixToCSVFile(labelMatrixTrain,"Label Matrix Train.csv")
    print ("Training features extracted")
    #print (len(featureMatrixTrain))
    
    #read Test data
    if USE_TEST:
        dataTest, labelTest = parse_testing(DIR_TEST) 
        print ("Test data read")  
        featureMatrixTest, labelMatrixTest, tweetMatrixTest, feature_names = featureExtraction(dataTest, labelTest, model)
        print ("Test features extracted")
    
    #print (featureMatrixTrain)
    #print labelMatrixTrain
    # split Train dataset into training and testing
    
    featureMatrixTrainSplitted, featureMatrixTestSplitted, labelMatrixTrainSplitted, labelMatrixTestSplitted = train_test_split(featureMatrixTrain, labelMatrixTrain, test_size=0.2, random_state=1)

    
    # train model
    if (CLASSIFIER=="SVM"):
        model = SVMclassifierTrain(featureMatrixTrainSplitted,labelMatrixTrainSplitted,featureMatrixTestSplitted,labelMatrixTestSplitted)
    if (CLASSIFIER=="DecTree"):
        model = DecisionTreeTrain(featureMatrixTrain,labelMatrixTrain)
    if (CLASSIFIER=="GaussProc"):
        model = GaussianProcessTrain(featureMatrixTrain,labelMatrixTrain)
    if (CLASSIFIER=="AdaBoost"):
        model = AdaBoostTrain(featureMatrixTrain,labelMatrixTrain)
    if (CLASSIFIER=="RandForest"):
        model = RandomForestTrain(featureMatrixTrain,labelMatrixTrain)
    if (CLASSIFIER=="NeuroNet"):
        model = NeuroNetTrain(featureMatrixTrain,labelMatrixTrain)
    if (CLASSIFIER=="Naive"):
        model = NaiveBayesTrain(featureMatrixTrain,labelMatrixTrain)
    
    # classify Train Testset
    #predictedTrain = classifierPredict(featureMatrixTestSplitted,model)
    #print ("Accuracy (Testset Train): "+str(scikitm.accuracy_score(labelMatrixTestSplitted,predictedTrain)))
    predictedTrain = classifierPredict(featureMatrixTrain,model)
    #print (len(predictedTrain))
    print ("Accuracy (Testset Train): "+str(scikitm.f1_score(labelMatrixTrain,predictedTrain, pos_label=1)))
    #saveTweetToCSVFile(predictedTrain,"predicted.txt")
    #saveTweetToCSVFile(dataLabel,"goldenTaskB.txt")
    # classify test set
    if USE_TEST:
        predictedTest = classifierPredict(featureMatrixTest,model)
        print ("Accuracy (Testset): "+str(scikitm.f1_score(labelMatrixTest,predictedTest, pos_label=1)))
        TASK = "A" # Define, A or B
        FNAME = './predictions-task' + TASK + '.txt'
        PREDICTIONSFILE = open(FNAME, "w")
        for p in predictedTest:
            PREDICTIONSFILE.write("{}\n".format(p))
        PREDICTIONSFILE.close()
        #savePredToFile(tweetMatrixTest,predictedTest,"predictedATest.json")
        #saveTweetToCSVFile(predictedTest,"featureMatrixTweet.txt")
        #f_importances(model.coef_,feature_names)
        #print(model._get_coef)
