# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:58:43 2017

@author: dadangewp
"""

'''
description: implementation of feature selection, preprocessing and scaling
'''
from features import userBased, contentBased
from features import featureEvaluation
import sklearn.preprocessing as pp
from iodata.saveToFile import saveMatrixToCSVFile
from iodata.saveToFile import saveTweetToCSVFile
import numpy as np
import re
from collections import Counter
from scipy.sparse import hstack, csr_matrix

def featureExtraction(data,gt,model): 
    
    featureMatrix = []
    featurePrint = []
    labelMatrix = [] 
    tweetMatrix = []
    combinedMatrix = []
    features = [#"Tweet",
                "emojiIncongruity",
                #"polarityContrastLIWC",
                #"polarityContrastEmolex",
                #"polarityContrastAfinn",
                #"SwearWords",
                #"Source-Reply-Sim",
                #"retweets",
                #"supportw",
                #"commentw",
                #"negation",
                #"denywordcount",
                #"questMarkCount",
                #"swearWord",
                #"repeatedChar",
                #"questMark",
                #"exclMarkCount",
                "hashTagsPresence",
                "hashTagsCount",
                "mentionCount",
                #"newLine",
                #"mentionFiltered",
                #"hasDotDot",
                #"numberDotDot",
                #"numberOfColon",
                #"hasQuote",
                "ConjunctionCount",
                #"PrepCount",
                #"upperCase",
                "textLengh",
                "LinkCount",
                "LinkPresence",
                #"emojiCount",
                #"emojiPresence",
                #"emoticonCount",
                #"resolveURL",
                #"sharedLink",
                "ironicWords",
                #"emojiSentiment",
                #"unweightMaxSim",
                #"unweightMinSim",
                #"unweightMaxDis",
                #"unweightMinDis",
                #"weightMaxSim",
                #"weightMinSim",
                #"weightMaxDis",
                #"weightMinDis"]
                #"capitalWordCount"]
                #"countAdjective",
                "countNouns",
                #"interjectionWord",
                #"countVerbs",
                #"Reply-To-Sim",
                #"TweetHierarchy",
                #"AFINN_Sentimen",
                "DAL_Pleasantness",
                "DAL_Activation",
                #"DAL_Imagery",
                #"ANEW_Pleasantness"]
                #"ANEW_Arrousal",
                #"ANEW_Dominance"]
                #"EmolexSurprise",
                #"EmolexAnger"]
                "EmolexTrust",
                #"EmolexPositive",
                "EmolexNegative",
                #"EmolexAnticipation",
                #"EmolexJoy",
                "EmolexFear",
                #"EmolexDisgust"]
                #"EmolexSadness"]
                #"LIWC_Pos",
                #"LIWC_Neg",
                #"EmoSNAnger",
                "EmoSNDisgust"]
                #"EmoSNFear",
                #"EmoSNJoy",
                #"EmoSNSad"]
                #"EmoSNSurprise",
                #"sentimentScore"]
    replyToTweetText = "-"
    clients = list()
    index = 0
    for tweet in data:
                #creating class labels
                label = gt[index]
                    
                
                #tweet console output
                '''
                print PPSourceTweetContent
                print PPTweetContent
                print tweetContent["source"]
                
                if label == "deny":
                    try:
                        print tweetContent["text"]
                    except:
                        print "error"
                        
                if(label=="deny"):
                        try:
                            print str(tweet)+": "+ sourceTweetContent["text"] + "::==::" + tweetContent["text"]
                        except:
                            print "print not possible"
                '''
                splittedTweet = set([]) 
                copiedTweet = tweet
                copiedTweet = re.sub(r'[^\w\s]','',copiedTweet)
                #print(copiedTweet)
                tweetToken = copiedTweet.split(" ")
                #print (tweetToken)
                for word in tweetToken:
                    word = word.lower()
                    #print(word)
                    splittedTweet.add(word)
                if label != "":
                #if label == "query":
                    labelMatrix.append(label)
                    tweetMatrix.append(tweet) 
                    #creating feature vector    
                    featureVector = ([#str(tweetContent["text"]),
                                          contentBased.emojiIncongruity(tweet),
                                          #contentBased.polarityShiftAFINN(tweet),
                                          #contentBased.polarityContrastLIWC(tweet),
                                          #contentBased.polarityContrastEmolex(tweet),
                                          #userBased.userFollowers(tweetContent["user"]),
                                          #contentBased.swearWords(tweet),
                                          #contentBased.textSimToSource([PPSourceTweetContent,PPTweetContent]),
                                          #contentBased.retweetCount(tweetContent),
                                          #contentBased.supportWordsCount(PPTweetContent),
                                          #contentBased.commentWordsCount(PPTweetContent),
                                          #contentBased.denyWordsCount(splittedTweet),
                                          #contentBased.questionMarksCount(tweet),
                                          #contentBased.negationWordsCount(tweet),
                                          #contentBased.getSwearCount(tweet),
                                          #contentBased.repeatedChar(tweet),
                                          #contentBased.questionMark(tweet),
                                          #contentBased.exclamationMarks(tweet),
                                          contentBased.hashTagsPresence(tweet),
                                          contentBased.hashTagsCount(tweet),
                                          contentBased.mentionMarks(tweet),
                                          #contentBased.newLineMark(tweet),
                                          #contentBased.mentionMarksFiltered(tweet),
                                          #contentBased.hasDotDotDot(tweet),
                                          #contentBased.numberOfDotDotDot(tweet),
                                          #contentBased.colonCount(tweet),
                                          #contentBased.getQuoteCount(tweet),
                                          contentBased.countConjunction(tweet),
                                          #contentBased.countPreposition(tweet),
                                          #contentBased.upperCaseCount(tweet),
                                          contentBased.textLenght(tweet),
                                          contentBased.linksCount(tweet),
                                          contentBased.linkPresence(tweet),
                                          #contentBased.getEmojiPresence(tweet),
                                          #contentBased.getEmojiCount(tweet),
                                          #contentBased.emoticonCount(tweet),
                                          #contentBased.resolveURL(tweet)
                                          #contentBased.sharedLink(splittedTweet),
                                          contentBased.ironicWordsCount(splittedTweet),
                                          #contentBased.getEmojiSentiment(tweet)
                                          #contentBased.unweightMaxSim(tweet,model),
                                          #contentBased.unweightMinSim(tweet,model),
                                          #contentBased.unweightMaxDis(tweet,model),
                                          #contentBased.unweightMinDis(tweet,model),
                                          #contentBased.weightMaxSim(tweet,model),
                                          #contentBased.weightMinSim(tweet,model),
                                          #contentBased.weightMaxDis(tweet,model),
                                          #contentBased.weightMinDis(tweet,model)
                                          #contentBased.capitalWordCount(tweet)
                                          #contentBased.countAdjective(tweet),
                                          contentBased.countNoun(tweet),
                                          #contentBased.interjectionWord(splittedTweet),
                                          #contentBased.countVerbs(tweet),
                                          #contentBased.textSimToSource([PPReplyToTweetContent,PPTweetContent]),
                                          #contentBased.tweetLevel(sourceTweet,tweet,hierarchy),
                                          #contentBased.getAffectiveAFINN(tweet),
                                          contentBased.getAffectiveDALPleasantness(tweet),
                                          contentBased.getAffectiveDALActivation(tweet),
                                          #contentBased.getAffectiveDALImagery(tweet),
                                          #contentBased.getAffectiveANEWPleasantness(tweet)
                                          #contentBased.getAffectiveANEWArrousal(tweet)
                                          #contentBased.getAffectiveANEWDominance(tweet)
                                          #contentBased.getSurpriseEmotionEmolex(tweet),
                                          #contentBased.getAngerEmotionEmolex(tweet)
                                          contentBased.getTrustEmotionEmolex(tweet),
                                          #contentBased.getPositiveEmotionEmolex(tweet),
                                          contentBased.getNegativeEmotionEmolex(tweet),
                                          #contentBased.getAnticipationEmotionEmolex(tweet),
                                          #contentBased.getJoyEmotionEmolex(tweet),
                                          contentBased.getFearEmotionEmolex(tweet),
                                          #contentBased.getDisgustEmotionEmolex(tweet)
                                          #contentBased.getSadnessEmotionEmolex(tweet)
                                          #contentBased.getAffectiveLIWCPos(tweet),
                                          #contentBased.getAffectiveLIWCNeg(tweet),
                                          #contentBased.getAngerESN(tweet)
                                          contentBased.getDisgustESN(tweet)
                                          #contentBased.getFearESN(tweet),
                                          #contentBased.getJoyESN(tweet),
                                          #contentBased.getSadESN(tweet)
                                          #contentBased.getSurpriseESN(tweet),
                                          #contentBased.isQuestion(tweetContent["text"])
                                          #contentBased.mentionMarks(tweetContent["text"])
                                          #contentBased.sentimentScore(tweet)
                                          #contentBased.sentimentSimilarity(sourceTweetContent["text"],tweetContent["text"])
                                        ])
                    #array = contentBased.addVector()
                    #for value in array:
                    #    featureVector.append(value)
                    featureMatrix.append(featureVector[:len(features)]) #number of features
                    featurePrint.append(tweet)
                    #print (featureVector)
                    #print("\n")
                    featureVector.append(label)
                    #featureVector.append(str(tweet))
                    combinedMatrix.append(featureVector)  
                    index = index + 1
 
    #print ("features:")                
    #print (features) 
    
    nrClients = Counter(clients)
                                                
    # standardization (zero mean, variance of one)
    stdScale = pp.StandardScaler().fit(featureMatrix)
    featureMatrixScaled = stdScale.transform(featureMatrix)
    
    #file output
    saveMatrixToCSVFile(featureMatrix,"featureMatrix.csv")
    saveTweetToCSVFile(labelMatrix,"tweetTextTrain.txt")
    #saveTweetToCSVFile(labelMatrix,"LabelMatrixTrain.txt")
    #print ("the size is :" + str(len(featurePrint)))
    #saveMatrixToCSVFile(featureMatrixScaled,"featureScaleMatrix.csv")
    #saveMatrixToCSVFile(labelMatrix,"labelMatrix.csv")
    #saveMatrixToCSVFile(combinedMatrix,"featureLabelMatrix.csv")
     
    #featureEvaluation.featureClassCoerr(featureMatrix,labelMatrix) 
    #print(features)
                              
    #return pp.normalize(featureMatrixScaled), labelMatrix, tweetMatrix                    
    return featureSelection(featureMatrixScaled), labelMatrix, tweetMatrix, features        

def featureSelection(featureMatrix):
    return featureMatrix