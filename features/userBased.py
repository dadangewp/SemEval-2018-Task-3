'''
description: implementations of user-based features
'''
import datetime as dt
import time as time

def userRegTime(user):
    #date = dt.datetime.strptime(user.created_at, '%a %b %d %H:%M:%S %z %y')
    return int((time.strftime("%Y"))) - int(user["created_at"][-4:user["created_at"].__len__()])
    
def userFollowers(user):
    return user["followers_count"]

def userFriends(user):
    return user["friends_count"]

def userInteractions(user,tweets):
    count = 0
    
    for tweet, tweetContent in tweets.items():
        if tweetContent["user"]["name"]==user["name"]:
            count += 1
    
    return count-1 #because self matching