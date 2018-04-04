# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 22:47:46 2018

@author: dadangewp
"""

import codecs
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class EmojiSentiment(object):

    emosent={}

    def __init__(self):
        self.emosent = {}
        #http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
        file=codecs.open('D:/PhD/RumourEval/Small Project on Stance Detection in Rumour/affectiveResources/emojisentiment.txt', encoding='UTF-8')
        for line in file:
            word, score = line.strip().split(' ')
            self.emosent[word] = float(score)

        self.pattern_split = re.compile(r"\W+")

        return

    def get_emoji_sentiment(self,text):

        sentiments=0.0
        words = self.pattern_split.split(text.lower())
        for word in words:
            if word in self.emosent:
                sentiments+=self.emosent[word]

        sentiments = format(sentiments, '.2f')
        sentiments = float(sentiments)
        return sentiments


if __name__ == '__main__':
    emosent = EmojiSentiment()
    text = ("@MohammadFarooq_ Asalamoalekum... :smiling_face_with_open_mouth: kesay mizaj ? Thank you so much. It was posted on 2nd December 2013. Already there. :smiling_face_with_halo:")
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",text).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(cleanedTweet)
    print(ss)
    sentiment = emosent.get_emoji_sentiment(cleanedTweet)
    print(sentiment)