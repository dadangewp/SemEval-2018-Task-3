# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:58:05 2017

@author: dadangewp
"""

import codecs
import re
from nltk.stem.porter import PorterStemmer

class Emoji(object):

    emoji=[]

    def __init__(self):
        self.emoji = []
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('D:/PhD/RumourEval/Small Project on Stance Detection in Rumour/affectiveResources/emojiFiltered.txt', encoding='UTF-8')
        for line in file:
            word = line.strip("\r\n")
            word = str(word)
            self.emoji.append(word.lower())
        #print(self.liwcpos)    
        self.pattern_split = re.compile(r"\W+")
        return

    def getEmojiCount(self,text):
        
        counter=0
        #print (words)
        #print (self.liwcpos)
        for word in self.emoji:
            count = len(re.findall(word, text))
            counter = counter + count


        return counter


if __name__ == '__main__':
    emoji = Emoji()
    emojicount=emoji.getEmojiCount("I enjoy stealing my husband hat every once in awhile :smiling_face_with_heart-shaped_eyes::face_throwing_a_kiss::flushed_face: #heyyall #GoodMorning http://t.co/LknoOm7VSb")
    print(emojicount)