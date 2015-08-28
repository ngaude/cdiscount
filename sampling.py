#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.externals import joblib
import time
import pandas as pd
import random
import sys

from utils import ddir,header

ext = '.0' # default value

print '-'*50

import sys
if len(sys.argv)==2:
    ext = '.'+str(int(sys.argv[1]))
    print 'sampling auto '+ext

def training_sample_random(df,N = 200,mincount=7):
    N = int(N)
    cl = df.Categorie3
    cc = cl.groupby(cl)
    s = (cc.count() >= mincount)
    labelmaj = s[s].index
    print 'sampling =',N,'samples for any of',len(labelmaj),'classes'
    dfs = []
    for i,cat in enumerate(labelmaj):
        if i%100==0:
            print 'sampling',i,'/',len(labelmaj),':'
        dfcat = df[df.Categorie3 == cat]
        sample_count = N
        if len(dfcat)>=sample_count:
            # undersample sample_count samples : take the closest first
            rows = random.sample(dfcat.index, sample_count)
            dfs.append(dfcat.ix[rows])
        else:
            # sample all samples + oversample the remaining
            dfs.append(dfcat)
            dfcat = dfcat.iloc[np.random.randint(0, len(dfcat), size=sample_count-len(dfcat))]
            dfs.append(dfcat)
    dfsample = pd.concat(dfs)
    dfsample = dfsample.reset_index(drop=True)
    dfsample = dfsample.reindex(np.random.permutation(dfsample.index),copy=False)
    return dfsample

##########################
# sampling a training set
##########################

df = pd.read_csv(ddir+'training_head.csv',sep=';',names = header()).fillna('')
fname = ddir+'training_sample.csv'+ext
print '>>'+fname
dfsample = training_sample_random(df,N=456,mincount=7)
dfsample.to_csv(fname,sep=';',index=False,header=False)
print '<<'+fname

##########################
# sampling a validation set
##########################

df = pd.read_csv(ddir+'training_tail.csv',sep=';',names = header()).fillna('')
fname = ddir+'validation_sample.csv'+ext
print '>>'+fname
dfsample = training_sample_random(df,N=7,mincount=1)
dfsample.to_csv(fname, sep=';',index=False,header=False)
print '<<'+fname

print '-'*50
