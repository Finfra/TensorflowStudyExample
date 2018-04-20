#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
def binEnc(n,staticLength=0):
    encodedList=[]
    while(n!=0):
        i=n%2
        encodedList.append(i)
        n=n>>1
    diff=staticLength-len(encodedList)
    encodedList.reverse()
    if(diff>0):
        encodedList=[0]*diff+encodedList
    return encodedList

print binEnc(65535)
print binEnc(65535,20)

    
