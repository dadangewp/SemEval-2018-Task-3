# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:50:56 2017

@author: dadangewp
"""

import codecs

def parse_training(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    #no = 0
    #clash = 0
    #other = 0
    #situational = 0
    corpus = []
    with codecs.open(fp, encoding="utf8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
    #            if label == 0:
    #                clash = clash + 1
    #            elif label == 1:
    #                other = other + 1
    #            elif label == 2:
    #                situational = situational + 1
    #            else :
    #                no = no + 1
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)
#    print ("training by clash:" + str(clash))
#    print ("training other:" + str(other))
#    print ("training situational:" + str(situational))
#    print ("training not irony:" + str(no))
    return corpus, y

def parse_testing(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    #no = 0
    #clash = 0
    #other = 0
    #situational = 0
    corpus = []
    with codecs.open(fp, encoding="utf8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
     #           if label == 0:
     #              clash = clash + 1
     #           elif label == 1:
     #               other = other + 1
     #           elif label == 2:
     #               situational = situational + 1
     #           else :
     #               no = no + 1
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)
#    print ("training by clash:" + str(clash))
#    print ("training other:" + str(other))
#    print ("training situational:" + str(situational))
#    print ("training not irony:" + str(no))
    return corpus, y