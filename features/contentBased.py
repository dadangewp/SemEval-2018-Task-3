'''
description: implementations of content-based features
'''
from collections import Counter
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re
import requests
import numpy as np
import datetime as dt
from features.wvBased import WVBased
from nltk import word_tokenize, pos_tag
from nltk.parse import stanford
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from affective.linguisticResourceDAL import DAL
from affective.lingusticResourceANEW import ANEW
from affective.linguisticResourceAFINN import AFINN
from affective.linguisticResourceLIWCPos import LIWCPos
from affective.linguisticResourceLIWCNeg import LIWCNeg
from emotion.emotionEmolex import Emolex
from emotion.emotionEmoSenticNetAnger import ESNAnger
from emotion.emotionEmoSenticNetDisgust import ESNDisgust
from emotion.emotionEmoSenticNetFear import ESNFear
from emotion.emotionEmoSenticNetJoy import ESNJoy
from emotion.emotionEmoSenticNetSad import ESNSad
from irony.emojiExtractor import Emoji
from irony.emojiSentiment import EmojiSentiment
from irony.swearWordExtractor import Swear
from emotion.emotionEmoSenticNetSurprise import ESNSurprise

parser = stanford.StanfordParser(model_path="D:\PhD\RumourEval\Small Project on Stance Detection in Rumour\stanford-parser-full-2017-06-09\model\englishPCFG.ser.gz")
#model = gensim.models.Word2Vec.load('brown_model')
dal = DAL()
sid = SentimentIntensityAnalyzer()
anew = ANEW()
afinn = AFINN()
liwcpos = LIWCPos()
liwcneg = LIWCNeg()
emolex = Emolex()
esnanger = ESNAnger()
esndisgust = ESNDisgust()
esnfear = ESNFear()
esnjoy = ESNJoy()
esnsad = ESNSad()
esnsurprise = ESNSurprise()
emoji = Emoji()
wvbased = WVBased()
emosent = EmojiSentiment()
swear = Swear()

def textSimToSource(tweetTexts):
      
    jaccard = float(len(tweetTexts[0].intersection(tweetTexts[1]))/float(len(tweetTexts[0].union(tweetTexts[0])))) 
    
    # count word occurrences
    #a_vals = Counter(tweetTexts[0])
    #b_vals = Counter(tweetTexts[1])

    # convert to word-vectors
    #words  = list(a_vals.keys() | b_vals.keys())
    #a_vect = [a_vals.get(word, 0) for word in words]        # [0, 0, 1, 1, 2, 1]
    #b_vect = [b_vals.get(word, 0) for word in words]        # [1, 1, 1, 0, 1, 0]

    # find cosine
    #len_a  = sum(av*av for av in a_vect) ** 0.5             # sqrt(7)
    #len_b  = sum(bv*bv for bv in b_vect) ** 0.5             # sqrt(4)
    #dot    = sum(av*bv for av,bv in zip(a_vect, b_vect))    # 3
    #cosine = dot / (len_a * len_b)  
    return jaccard

def getAffectiveDALActivation(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, activation, imagery,pleasantness_sum, activation_sum, imagery_sum=dal.get_dal_sentiment(cleanedTweet)
    return activation

def getAffectiveDALPleasantness(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, activation, imagery,pleasantness_sum, activation_sum, imagery_sum=dal.get_dal_sentiment(cleanedTweet)
    return pleasantness

def getAffectiveDALImagery(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, activation, imagery,pleasantness_sum, activation_sum, imagery_sum=dal.get_dal_sentiment(cleanedTweet)
    return imagery

def getAffectiveANEWPleasantness(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, arrousal, dominance, pleasantness_sum, arrousal_sum, dominance_sum=anew.get_anew_sentiment(cleanedTweet)
    return pleasantness

def getAffectiveANEWArrousal(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, arrousal, dominance, pleasantness_sum, arrousal_sum, dominance_sum=anew.get_anew_sentiment(cleanedTweet)
    return arrousal

def getAffectiveANEWDominance(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    pleasantness, arrousal, dominance, pleasantness_sum, arrousal_sum, dominance_sum=anew.get_anew_sentiment(cleanedTweet)
    return dominance

def getAffectiveAFINN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=afinn.get_afinn_sentiment(cleanedTweet)
    if sentiment > 0:
        return 1
    else:
        return 0

def getAffectiveLIWCPos(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcpos.get_liwcpos_sentiment(cleanedTweet)
    return sentiment

def getAffectiveLIWCNeg(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentiment=liwcneg.get_liwcneg_sentiment(cleanedTweet)
    return sentiment

def getSurpriseEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("surprise")
    return count

def getAngerEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("anger")
    return count

def getAnticipationEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("anticipation")
    return count

def getDisgustEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("disgust")
    return count

def getFearEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("fear")
    return count

def getJoyEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("joy")
    return count

def getSadnessEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("sadness")
    return count

def getPositiveEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("positive")
    return count

def getNegativeEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("negative")
    return count

def getTrustEmotionEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    surprise = emolex.get_emotion(cleanedTweet)
    count = 0
    for lst in surprise:
        count = count + lst.count("trust")
    return count

def getAngerESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esnanger.get_esnanger_sentiment(cleanedTweet)
    return score

def getDisgustESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esndisgust.get_esndisgust_sentiment(cleanedTweet)
    return score

def getFearESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esnfear.get_esnfear_sentiment(cleanedTweet)
    return score

def getJoyESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esnjoy.get_esnjoy_sentiment(cleanedTweet)
    return score

def getSadESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esnsad.get_esnsad_sentiment(cleanedTweet)
    return score

def getSurpriseESN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = esnsurprise.get_esnsurprise_sentiment(cleanedTweet)
    return score

def getEmojiCount(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = emoji.getEmojiCount(cleanedTweet)
    return score

def getEmojiPresence(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = emosent.get_emoji_sentiment(cleanedTweet)
    #print (score)
    if(score != 0):
        return 1
    else :
        return 0

def getEmojiSentiment(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = emosent.get_emoji_sentiment(cleanedTweet)
    return score
    #print (score)
    #if(score > 0):
    #    return 1
    #else :
    #    return 0

def getSwearCount(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = swear.getSwearCount(tweetText)
    return score

def polarityContrastLIWC(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    pos = liwcpos.get_liwcpos_sentiment(cleanedTweet)
    neg = liwcneg.get_liwcneg_sentiment(cleanedTweet)
    if(pos != 0 and neg != 0):
        return 1
    else :
        return 0

def polarityContrastEmolex(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    pos = getPositiveEmotionEmolex(cleanedTweet)
    neg = getNegativeEmotionEmolex(cleanedTweet)
    if(pos != 0 and neg != 0):
        return 1
    else :
        return 0

def repeatedChar(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    repeat = len(re.findall(r'((\w)\2{2,})', cleanedTweet))
    if repeat > 0:
        return 1
    else :
        return 0

def emojiIncongruity(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = getEmojiSentiment(tweetText)
    textsent = sentimentScore(cleanedTweet)
    distance = textsent - score
    if score != 0.0:
        #if textsent == 0.0:
        #    return 0
        if (score < 0 and textsent > 0) or (score > 0 and textsent < 0):
            return 1
        elif (score > 0 and textsent > 0) and (abs(score-textsent)>0.5):
            return 1
        elif (score < 0 and textsent < 0) and (abs(score-textsent)>0.5):
            return 1
        else:
            return 0
    else:
        return 0
    #if score == 0.0:
    #    return 0
    #else:
    #return distance

def polarityShiftAFINN(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    cleanedTweet.lower()
    score = afinn.getAfinnShift(cleanedTweet)
    return score

def retweetCount(tweet): 
    return tweet["retweet_count"]

def avgWordLength(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentenceLength = len(cleanedTweet)
    tweetTokenList = tokenizer(cleanedTweet)
    wordCount = len(tweetTokenList)
    if (sentenceLength > 0 and wordCount > 0):
        avg = float(sentenceLength/wordCount)
        avg = format (avg,'.2f')
        return float(avg)
    else :
        return 0 

def countNoun(tweetText):
    score = sum(1 for word, pos in pos_tag(word_tokenize(tweetText)) if pos.startswith('NN'))
    return score

def countAdjective(tweetText):
    score = sum(1 for word, pos in pos_tag(word_tokenize(tweetText)) if pos.startswith('JJ'))
    return score

def countVerbs(tweetText):
    score = sum(1 for word, pos in pos_tag(word_tokenize(tweetText)) if pos.startswith('V'))
    return score

def countConjunction(tweetText):
    score = sum(1 for word, pos in pos_tag(word_tokenize(tweetText)) if pos.startswith('IN'))
    return score

def countPreposition(tweetText):
    score = sum(1 for word, pos in pos_tag(word_tokenize(tweetText)) if pos.startswith('CC'))
    return score

def supportWordsCount(tweetText):
    SUPPORTWORDS = set(["clarifi", "right", "evid", "confirm", "support", "definit", "discov", "explain", "truth", "true", "offici"])
    #tweetTokenList = tokenizer(tweetText) 
    #return len([token for token in SUPPORTWORDS if token in tweetTokenList])
    return len(tweetText.intersection(SUPPORTWORDS))

def commentWordsCount(tweetText):
    COMMENTWORDS = set(["comment", "claim", "accord", "sourc", "show", "captur", "say", "report", "observ", "footag"])
    return len(tweetText.intersection(COMMENTWORDS))

def emoticonCount(tweetText):
    emoticons = [":)",":p",":P"]
    count = 0
    for emot in emoticons:
        if emot in tweetText:
            count = count + 1
    return count

def capitalWordCount(tweetText):
    num = 0
    for char in tweetText:
        if char.isupper():
            num+=1
    #print (num)
    return num

def ironicWordsCount(tweetText):
    ironicWords = set(["love","lovely","like","great","brilliant","perfect","thanks","glad"])
    return len(tweetText.intersection(ironicWords))

def sharedLink(tweetText):
    sharedLink = set(["via","visit"])
    return len(tweetText.intersection(sharedLink))

def interjectionWord(tweetText):
    interjects = set(["uh","oh","yeah"])
    return len(tweetText.intersection(interjects))

def resolveURL(tweetText):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweetText)
    #print(tweetText)
    flag = 0
    for url in urls:
        link = ""
        try:
            link = requests.get(url, verify=False, timeout=10).url
        except :
            #print ("connection refused")
            link = "error"     
        link = str(link)
        #print (link)
        if "twitter.com" in link:
            flag = 1
        elif "instagram.com" or "youtube.com" in link :
            flag = 2
        else :
            continue

    if flag == 1:
        return 1
    elif flag == 2:
        return 2
    else :
        return 0

def denyWordsPresence(text):
    SwearWords = ["smh","bs","li","lie","ly","lol","bullshit","wtf","fuck","idiot","aw"]
    
    count = len(text.intersection(SwearWords)) 

    if (count > 0):
        return 1
    else:
        return 0

def denyWordsCount(text):
    SwearWords = ["smh","bs","li","lie","ly","lol","bullshit","wtf","fuck","idiot","aw"]
    
    count = len(text.intersection(SwearWords)) 

    return count

def questionWordsCount(tweetText):
    QUESTIONWORDS = set(["what", "who", "question", "whi", "how", "where", "wonder"])
    return len(tweetText.intersection(QUESTIONWORDS))

def questionMarksCount(tweetText):
    return len(re.findall("\?", tweetText))

def colonCount(tweetText):
    count = len(re.findall(":", tweetText))
    if (count > 0):
        return 1
    else :
        return 0
    
def upperCaseCount(tweetText):
    count = 0
    for i in tweetText:
        if i.isupper():
            count = count + 1
    #print (count)
    if count > 0:
        return 1
    else : 
        return 0

def questionMark(tweetText):
    count = len(re.findall("\?", tweetText))
    if (count > 0):
        return 1
    else:
        return 0

def newLineMark(tweetText):
    count = len(re.findall("\|", tweetText))
    if (count > 0):
        return 1
    else:
        return 0
    
def exclamationMarks(tweetText):
    count = len(re.findall("\!\!\!", tweetText))
    if count > 0:
        return 1
    else:
        return 0

def mentionMarks(tweetText):
    count = len(re.findall("\@", tweetText))
    if count == 0:
        return 0
    else :
        return 1

def mentionMarksFiltered(tweetText):
    count = len(re.findall("\@", tweetText))
    if count == 0:
        if len(tweetText) < 100 or len(tweetText) > 125:
            return 0
        else:
            return 1
    else :
        return 0

def getQuoteCount(tweetText):
    matches1 = len(re.findall(r'\"(.+?)\"',tweetText))
    matches2 = len(re.findall(r'\'(.+?)\'',tweetText))
    if (matches1 > 0 or matches2 > 0):
        return 1
    else:
        return 0

def hasDotDotDot(tweetText):
    count = len(re.findall("...", tweetText))
    if count > 0:
        return 1
    else:
        return 0

def numberOfDotDotDot(tweetText):
    count = len(re.findall("\.", tweetText))
    if count > 0:
        return 1
    else:
        return 0

def textLenght(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetText).split())
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    tweetTokenList = tokenizer(cleanedTweet) 
    return len(tweetTokenList)

def linksCount(tweetText):
    return len(re.findall("http", tweetText))

def linkPresence(tweetText):
    if(len(re.findall("http", tweetText)) > 0):
        return 1
    else:
        return 0
    
def hashTagsCount(tweetText):
    count = len(re.findall("#", tweetText))
    if count == 0:
        return 0
    elif count > 3:
        return 2
    else:
        return 1
    #return count

def hashTagsPresence(tweetText):
    if(len(re.findall("#", tweetText)) > 0):
        return 1
    else:
        return 0
    
def swearWords(text):
    SwearWords = ["fuck","asshole","shit","ass","bitch","nigga","hell","whore","dick","pussy","slut","putta","damn","fag","cum","cunt","cock","blowjob","retard"]
    count = 0

    for w in SwearWords:
        count += len(re.findall(w, text))

    return count

def bowTokenizer(text):
    #nltk.download("stopwords")
    stopTerms = set(stopwords.words("english"))
    tokens = set(text.split()).difference(stopTerms)  
    return tokens

def tokenizer(text):   
    vectorizer = CountVectorizer(min_df=1)
    analyze = vectorizer.build_analyzer()
    return analyze(text)

def negationWordsCount(text):   
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",text).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sentences = parser.raw_parse(cleanedTweet)
    
    found = 0;
    #print (sentences)
    for line in sentences:
        for sentence in line:
            words = str (sentence)
            #print (words)
            if("neg" in words):
                found = 1
                #print ("found it")
                break

    if found == 1:
        return 1
    else :
        return 0
    
def stemmer(terms):
    
    stemmer = PorterStemmer()
    stemmedTerms = set([]) 
     
    try: 
        for term in terms:
            stemmedTerm = stemmer.stem(term)
            stemmedTerms.add(str(stemmedTerm))
        
        return stemmedTerms
    except:
        return terms
    
def replyTimeToSource(sourceDate,replyDate):
    
    sourceTime = dt.datetime.strptime(sourceDate[:sourceDate.__len__()-11],'%a %b %d %H:%M:%S')
    replyTime = dt.datetime.strptime(replyDate[:replyDate.__len__()-11],'%a %b %d %H:%M:%S')
    timeDelta = replyTime - sourceTime
    replyTime = (timeDelta.days * 24 * 60) + (timeDelta.seconds/60)
    replyTime = format (replyTime,'.2f')
    replyTime = float(replyTime)
    return replyTime

    
def possiblySensitive(value):
    
    if value == "True": # no boolean -> text
        return 1
    else:
        return 0
    
def twitterClient(link):
    
    if link.find("Twitter Web Client") != -1: 
        return 1
    if link.find("Twitter for Mac") != -1: 
        return 1
    if link.find("Twitter for Websites") != -1: 
        return 1
    if link.find("Twitter for ") != -1: 
        return 0
    else:
        return 0
    
def tweetLevel(sourceId,targetId,hierarchy):
    
    count = 0
    tweethierachy = ""

    if sourceId == targetId:
        return 0
    else:
        for i, t in hierarchy.items():
            if int(i) == sourceId:
                tweethierachy = str(t).replace("u", " ").split()
                break
    
        for x in tweethierachy:
            if (x.find("{") != -1):
                count = count + x.count('{')
            if (x.find("}") != -1):
                count = count - x.count('}')
            if (x.find(str(targetId)) != -1):
                return count
    
    return 0
            
def isQuestion(tweetText):
    #print ("masuk")
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",tweetText).split())
    cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    listSentence = []
    indexDelimiter = 0;
    indexDelimiter = cleanedTweet.find("?")
    if indexDelimiter != 0:
        cleanedTweet = cleanedTweet[:indexDelimiter+1] +"|"+cleanedTweet[indexDelimiter+1:]
    listTweet = cleanedTweet.split("|")
    for tweet in listTweet:
        splitTweet = tweet.split(".")
        for t in splitTweet:
            listSentence.append(t);
    sentences = parser.raw_parse_sents(listSentence)
    
    found = 0;
    #print (sentences)
    for line in sentences:
        for sentence in line:
            words = str (sentence)
            #print (words)
            if("SQ" in words or "SBARQ" in words):
                found = 1
                #print ("found it")
                break

    if found == 1:
        return 1
    else :
        return 0

def sentimentScore(tweetText):
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetText).split())
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(cleanedTweet)
    if ss["compound"] > 0.6:
        return 1
    else :
        return 0
    
def avg_feature_vector(tweetText):
    #function to average all words vectors in a given paragraph 
    featureVec = np.zeros((100,), dtype="float32")
    nwords = 0
    cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetText).split())
    cleanedTweet.replace("'","")
    cleanedTweet.replace('"',"")
    cleanedTweet.replace('/',"")
    cleanedTweet.replace("\\","")
    words = tokenizer(cleanedTweet)
    for word in words:
        featureVec = np.add(featureVec, model[word])
        nwords = nwords+1
    if(nwords>0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def addVector():
    vector = np.full((3,1),7)
    return vector

def sentimentSimilarity(source,reply):
    cleanedTweet1 = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",source).split())
    cleanedTweet1.replace("'","")
    cleanedTweet1.replace('"',"")
    cleanedTweet1.replace('/',"")
    cleanedTweet1.replace("\\","")
    cleanedTweet2 = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",reply).split())
    cleanedTweet2.replace("'","")
    cleanedTweet2.replace('"',"")
    cleanedTweet2.replace('/',"")
    cleanedTweet2.replace("\\","")
    sid = SentimentIntensityAnalyzer()
    ss1 = sid.polarity_scores(cleanedTweet1)
    ss2 = sid.polarity_scores(cleanedTweet2)
    if (ss1 == ss2):
        return 1
    else :
        return 0

def unweightMaxSim(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return a

def unweightMinSim(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return b

def unweightMaxDis(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return c

def unweightMinDis(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return d

def weightMaxSim(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return e

def weightMinSim(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return f

def weightMaxDis(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return g

def weightMinDis(tweet,model):
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(tweet,model)
    return h