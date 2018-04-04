# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 23:46:41 2018

@author: dadangewp
"""

import codecs
import re
from nltk.stem.porter import PorterStemmer

class Swear(object):

    swear=[]

    def __init__(self):
        self.swear = []
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('D:/PhD/RumourEval/Small Project on Stance Detection in Rumour/affectiveResources/SwearWord.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = str(word)
            self.swear.append(word.lower())
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def getSwearCount(self,text):
        
        counter=0
        #print (words)
        #print (self.liwcpos)
        for word in self.swear:
            count = len(re.findall(word, text))
            counter = counter + count


        return counter
    
    def getSwearList(self, text):
        swears = ""
        for word in self.swear:
            count = len(re.findall(word, text))
            if(count > 0):
                swears = swears +" "+ word
        #print (emojis)
        return swears


if __name__ == '__main__':
    swear = Swear()
    emojicount=swear.getSwearCount("I love fuck shit goddamn")
    print(emojicount)