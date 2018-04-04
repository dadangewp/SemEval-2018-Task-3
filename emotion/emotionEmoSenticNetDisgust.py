# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 01:15:56 2017

@author: dadangewp
"""

import codecs
import re
from nltk.stem.porter import PorterStemmer

class ESNDisgust(object):


    def __init__(self):
        self.esnDisgust = []
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('D:/PhD/RumourEval/Small Project on Stance Detection in Rumour/affectiveResources/esn/EmoSN_disgust.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = stemmer.stem(word)
            self.esnDisgust.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_esndisgust_sentiment(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        #words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        #print (words)
        #print (self.liwcpos)
        for word in words:
            stemmed = stemmer.stem(word)
            #print(stemmed)
            if stemmed in self.esnDisgust:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    esndisgust= ESNDisgust()
    sentiment=esndisgust.get_esndisgust_sentiment("yellow sticky")
    print(sentiment)