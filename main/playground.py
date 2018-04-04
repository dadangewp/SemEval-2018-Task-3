
import datetime as dt
import time as time

if __name__ == '__main__':
    
    #date1 = dt.datetime.strptime("Wed Mar 18 12:57:11 +0001 2009", '%a %b %d %H:%M:%S %+%0%0%0%0 %y')
    #date2 = dt.datetime.strptime("Wed Mar 18 13:52:11 +0001 2009", '%a %b %d %H:%M:%S %+%0%0%0%0 %y')
    
    d1 = dt.datetime.strptime("Wed Mar 18 12:57:11 +0001 2009"[:"Wed Mar 18 12:57:11 +0001 2009".__len__()-11],'%a %b %d %H:%M:%S')
    d2 = dt.datetime.strptime("Wed Mar 18 13:52:11 +0001 2009"[:"Wed Mar 18 12:57:11 +0001 2009".__len__()-11],'%a %b %d %H:%M:%S')
        
    diff = d2 -d1
    diff_minutes = (diff.days * 24 * 60) + (diff.seconds/60)

    print(diff_minutes)

    
    