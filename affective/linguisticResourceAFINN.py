# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:16:15 2017

@author: dadangewp
"""

import codecs
import re

class AFINN(object):

    afinn={}

    def __init__(self):
        self.afinn = {}
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('D:/PhD/RumourEval/Small Project on Stance Detection in Rumour/affectiveResources/Afinn_en.txt', encoding='UTF-8')
        for line in file:
            word, score = line.strip().split(' ')
            self.afinn[word] = float(score)

        self.pattern_split = re.compile(r"\W+")

        return

    def get_afinn_sentiment(self,text):

        sentiments=0.0
        words = self.pattern_split.split(text.lower())
        for word in words:
            if word in self.afinn:
                sentiments+=self.afinn[word]

        sentiments = format(sentiments, '.2f')
        sentiments = float(sentiments)
        return sentiments
    
    def getAfinnShift(self,text):
        countPos = 0
        countNeg = 0
        words = self.pattern_split.split(text.lower())
        for word in words:
            if word in self.afinn:
                if self.afinn[word] < 0 :
                    countNeg = countNeg + 1
                elif self.afinn[word] > 0 :
                    countPos = countPos + 1
                else : 
                    continue
        if countPos > 0 and countNeg > 0:
            return 1
        else :
            return 0


if __name__ == '__main__':
    afinn = AFINN()
    sentiment=afinn.get_afinn_sentiment("Every 28 hours a black male is killed in the United States by police or vigilantes this makes me cry")
    print(sentiment)