# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 01:36:21 2017

@author: dadangewp
"""

import codecs
import re
from nltk.stem.porter import PorterStemmer

class ESNFear(object):


    def __init__(self):
        self.esnFear = []
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('D:/PhD/RumourEval/Small Project on Stance Detection in Rumour/affectiveResources/esn/EmoSN_fear.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = stemmer.stem(word)
            self.esnFear.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_esnfear_sentiment(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        #words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        #print (words)
        #print (self.liwcpos)
        for word in words:
            stemmed = stemmer.stem(word)
            #print(stemmed)
            if stemmed in self.esnFear:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    esnfear = ESNFear()
    sentiment=esnfear.get_esnfear_sentiment("trouble anxiety")
    print(sentiment)