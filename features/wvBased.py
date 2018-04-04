# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 19:21:50 2017

@author: dadangewp
"""

import re
from sklearn.metrics.pairwise import cosine_similarity;
import warnings
import gensim

warnings.filterwarnings('ignore')

#model = dict(zip(w2v.wv.index2word, w2v.wv.syn0))

class WVBased (object):
    
    def f3(self, seq):
       # Not order preserving
       keys = {}
       for e in seq:
           keys[e] = 1
       return list(keys.keys())
    
    def get_distance(self, sentence):
        dic1 = {}
        words = sentence.split(' ')
        index = 0
        for word in words:
            dic1[word] = index
            index += 1
        return dic1
    
    
    def get_wv_features(self, line, model):
        cleanedTweet = ' '.join(re.sub("(@[A-Za-z0-9_-]+)|(#[A-Za-z0-9_-]+)|(^https?:\/\/.*[\r\n]*)"," ",line).split())
        cleanedTweet = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', cleanedTweet)
        cleanedTweet.replace("'","")
        cleanedTweet.replace('"',"")
        cleanedTweet.replace('/',"")
        cleanedTweet.replace("\\","")
        line = cleanedTweet.strip().lower()    
        contents = line.split('\t')
        sentence = contents[0]
        words = re.findall(r"[\w']+|[.,!?;]",sentence)
            
        max_same = []
        min_same = []
        max_weighted = []
        min_weighted = []
    
        word_list = self.f3(sentence.split(' '))
    
        dic1 = self.get_distance(sentence)
        
        
    
        for e1 in word_list:
            if e1 not in model.vocab:
                continue
            output = e1+"\t"
            same_list = []
            weighted_list = []
            for e2 in word_list:
                
                if e2 not in model.vocab:
                    continue
                    
                if e1 == e2:
             #   output += '0\t'
                    continue
                
                weighted_list.append(model.similarity(e1,e2)/(dic1[e1] - dic1[e2])*2)
                same_list.append(model.similarity(e1,e2))
                    
            if len(same_list) != 0:
                max_same.append (max(same_list))
                min_same.append(min(same_list))
            else:
                min_same.append(0)           
                max_same.append(0)
            
            if len(weighted_list) != 0:
                min_weighted.append(min(weighted_list))
                max_weighted.append(max(weighted_list))
            else:
                min_weighted.append(0)
                max_weighted.append(0)
        #print(min_same)  
              
       # min_same.append(0)
       # max_same.append(0)
       # min_weighted.append(0)
       # max_weighted.append(0)
    
        if len(max_same) == 0:
            a = 0
            b = 0
        else:
            a = max(max_same)
            b = min(max_same)
            
        if len(min_same) == 0:
            c = 0
            d = 0
        else:
            c = max(min_same)
            d = min(min_same)
            
        if len(max_weighted) == 0:
            e = 0
            fk = 0
        else:
            e = max(max_weighted)
            fk = min(max_weighted)
            
        if len(min_weighted) == 0:
            g = 0
            h = 0
        else:
            g = max(min_weighted)
            h = min(min_weighted)
            
       
        
        stt = ('13250:'+str(2-a)+' 13251:'+str(2-b)+
               ' 13252:'+str(2-c)+' 13253:'+str(2-d) +
              ' 13254:'+str(2-e)+' 13255:'+str(2-fk)+
             ' 13256:'+str(2-g)+' 13257:'+str(2-h)
            )
        #print (stt)
        return(float(a),float(b),float(c),float(d),float(e),float(fk),float(g),float(h))
   
if __name__ == '__main__':   
    wvbased = WVBased()
    line = "A man needs a woman like a fish needs bicycle"
    model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\dadangewp\\Stance Detection in Rumor on Social Media\\GoogleNews-vectors-negative300.bin', binary=True)
    a,b,c,d,e,f,g,h = wvbased.get_wv_features(line,model)
    #print (str(h))