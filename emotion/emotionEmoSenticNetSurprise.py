# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 01:08:10 2017

@author: dadangewp
"""

import codecs
import re
from nltk.stem.porter import PorterStemmer

class ESNSurprise(object):

    liwcpos=[]

    def __init__(self):
        self.esnSurprise = []
        stemmer = PorterStemmer()
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('D:/PhD/RumourEval/Small Project on Stance Detection in Rumour/affectiveResources/esn/EmoSN_surprise.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = stemmer.stem(word)
            self.esnSurprise.append(word)
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def get_esnsurprise_sentiment(self,text):
        
        stemmer = PorterStemmer()
        counter=0
        #words = self.pattern_split.split(text.lower())
        words = text.split(" ")
        #print (words)
        #print (self.liwcpos)
        for word in words:
            stemmed = stemmer.stem(word)
            #print(stemmed)
            if stemmed in self.esnSurprise:
                counter = counter + 1


        return counter


if __name__ == '__main__':
    esnsurprise= ESNSurprise()
    sentiment=esnsurprise.get_esnsurprise_sentiment("bedaze tingle ")
    print(sentiment)