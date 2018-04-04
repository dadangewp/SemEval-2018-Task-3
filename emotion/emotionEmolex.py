# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:51:08 2017

@author: dadangewp
"""

import codecs
import re

class Emolex(object):

    emolex={}

    def __init__(self):
        self.emolex = {}
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('D:/PhD/RumourEval/Small Project on Stance Detection in Rumour/affectiveResources/Emolex.txt', encoding='UTF-8')
        for line in file:
            word, emotion, relation = line.strip().split('\t')
            if(relation == "1"):
                if(word in self.emolex):
                    self.emolex[word].append(emotion)
                else:
                    self.emolex[word] = []
                    self.emolex[word].append(emotion)

        #self.pattern_split = re.compile(r"\W+")

        return

    def get_emotion(self,text):

        emotionList = []
        words = text.split(" ")
        for word in words:
            if word in self.emolex:
                emotionList.append(self.emolex[word])
        return emotionList


if __name__ == '__main__':
    emolex = Emolex()
    emotion=emolex.get_emotion("abandon abandoned")
    count = 0
    for lst in emotion:
        count = count + lst.count("negative")
    print (count)