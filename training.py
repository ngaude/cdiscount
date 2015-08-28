#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:14:18 2015

@author: ngaude
"""
##################
# NOTE NOTE NOTE #
##################

ext = '.0' # default value

print '-'*50


best_regularisation = {1000012160: 13, 1000010560: 67, 1000017026: 8103, 1000009409: 67, 1000001161: 8103, 1000005258: 13, 1000011479: 3, 1000000269: 1636, 1000002192: 67, 1000004369: 330, 1000010900: 1636, 1000013441: 3, 1000000025: 100, 1000012702: 8103, 1000001700: 330, 1000014375: 8103, 1000003924: 13, 1000013220: 1636, 1000009134: 330, 1000014006: 13, 1000016184: 8103, 1338: 13, 1000006204: 67, 1000010220: 8103, 1000000832: 8103, 193: 8103, 1000012099: 8103, 1000009673: 8103, 1000012491: 13, 1000001359: 1636, 1000010704: 67, 1000014956: 8103, 1000002514: 1636, 1000003923: 1636, 1000001876: 13, 1000009977: 330, 1000001188: 3, 1000015968: 100, 1000017380: 3, 1000000230: 100, 1000009575: 330, 1000000235: 8103, 1000003564: 330, 1000010096: 8103, 1000010356: 8103, 1000002677: 1636, 1000008694: 1636, 1000000247: 330, 1000004588: 100, 340: 13, 1000015738: 13, 1000016087: 67}

import sys
if len(sys.argv)==2:
    ext = '.'+str(int(sys.argv[1]))
    print 'training auto '+ext

from utils import wdir,ddir,header,normalize_file,add_txt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from os.path import basename

from utils import itocat1,itocat2,itocat3
from utils import cat1toi,cat2toi,cat3toi
from utils import cat3tocat2,cat3tocat1,cat2tocat1
from utils import cat1count,cat2count,cat3count
from utils import training_sample
from os.path import isfile

from joblib import Parallel, delayed
import time

def log_proba(df,vec,cla):
    assert 'txt' in df.columns
    X = vec.transform(df.txt)
    lp = cla.predict_log_proba(X)
    return (cla.classes_,lp)

def vectorizer_stage1(txt):
    vec = TfidfVectorizer(
        min_df = 1,
        stop_words = None,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    X = vec.fit_transform(txt)
    return (vec,X)

def vectorizer_stage3(txt):
    vec = TfidfVectorizer(
        min_df = 1,
        stop_words = None,
        smooth_idf=True,
        norm='l2',
        sublinear_tf=True,
        use_idf=True,
        ngram_range=(1,2))
    X = vec.fit_transform(txt)
    return (vec,X)

def training_stage1(dftrain,dfvalid):
    fname = ddir + 'joblib/stage1'+ext
    print '-'*50
    print 'training',basename(fname)
    df = dftrain
    dfv = dfvalid
    vec,X = vectorizer_stage1(df.txt)
    Y = df['Categorie1'].values
    cla = LogisticRegression(C=100)
    cla.fit(X,Y)
    labels = np.unique(df.Categorie1)
    Xv = vec.transform(dfv.txt)
    Yv = dfv['Categorie1'].values
    sct = cla.score(X[:10000],Y[:10000])
    scv = cla.score(Xv,Yv)
    joblib.dump((labels,vec,cla),fname)
    del X,Y,Xv,Yv,vec,cla
    return sct,scv

def training_stage3(dftrain,dfvalid,cat1,i):
    fname = ddir + 'joblib/stage3_'+str(cat1)+ext
    df = dftrain[dftrain.Categorie1 == cat1].reset_index(drop=True)
    dfv = dfvalid[dfvalid.Categorie1 == cat1].reset_index(drop=True)
    labels = np.unique(df.Categorie3)
    if len(labels)==1:
        joblib.dump((labels,None,None),fname)
        scv = -1
        sct = -1
        print 'training',cat1,'\t\t(',i,') : N=',len(df),'K=',len(labels)
        print 'training',cat1,'\t\t(',i,') : training=',sct,'validation=',scv
        return (sct,scv)
    vec,X = vectorizer_stage3(df.txt)
    Y = df['Categorie3'].values
    cla = LogisticRegression(C=best_regularisation.get(cat1,100))
    cla.fit(X,Y)
    labels = np.unique(df.Categorie3)
    sct = cla.score(X[:min(10000,len(df))],Y[:min(10000,len(df))])
    if len(dfv)==0:
        scv = -1
    else:
        Xv = vec.transform(dfv.txt)
        Yv = dfv['Categorie3'].values
        scv = cla.score(Xv,Yv)
    print 'training',cat1,'\t\t(',i,') : N=',len(df),'K=',len(labels)
    print 'training',cat1,'\t\t(',i,') : training=',sct,'validation=',scv
    joblib.dump((labels,vec,cla),fname)
    del vec,cla
    return (sct,scv)

#######################
# training
# stage1 : Categorie1 
# stage3 : Categorie3|Categorie1
#######################

dftrain = pd.read_csv(ddir+'training_sample.csv'+ext,sep=';',names = header()).fillna('')
dfvalid = pd.read_csv(ddir+'validation_sample.csv'+ext,sep=';',names = header()).fillna('')
dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')

add_txt(dftrain)
add_txt(dfvalid)
add_txt(dftest)

dftrain = dftrain[['Categorie3','Categorie1','txt']]
dfvalid = dfvalid[['Categorie3','Categorie1','txt']]
dftest = dftest[['Identifiant_Produit','txt']]


# training stage1

dt = -time.time()
sct,scv = training_stage1(dftrain,dfvalid)
dt += time.time()

print '##################################'
print '# stage1 elapsed time :',dt
print '# stage1 training score :',sct
print '# stage1 validation score :',scv
print '##################################'

# training parralel stage3
cat1 = np.unique(dftrain.Categorie1)
#training_stage3(ctx,cat1[0])
dfts = []
dfvs = []
for cat in cat1:
    dfts.append(dftrain[dftrain.Categorie1 == cat].reset_index(drop=True))
    dfvs.append(dfvalid[dfvalid.Categorie1 == cat].reset_index(drop=True))


dt = -time.time()
scs = Parallel(n_jobs=3)(delayed(training_stage3)(dft,dfv,cat,i) for i,(dft,dfv,cat) in enumerate(zip(dfts,dfvs,cat1)))
dt += time.time()

sct = np.median([s for s in zip(*scs)[0] if s>=0])
scv = np.median([s for s in zip(*scs)[1] if s>=0])

print '##################################'
print '# stage3 elapsed time :',dt
print '# stage3 training score :',sct
print '# stage3 validation score :',scv
print '##################################'

#######################
# predicting
#######################

# predict : 
# P(Categorie3) = P(Categorie1) *  P(Categorie3|Categorie1)


#######################
# stage 1 log proba filling
#######################
stage1_log_proba_valid = np.full(shape=(len(dfvalid),cat1count),fill_value = -666.,dtype = float)
stage1_log_proba_test = np.full(shape=(len(dftest),cat1count),fill_value = -666.,dtype = float)

fname = ddir + 'joblib/stage1'+ext
(labels,vec,cla) = joblib.load(fname)
(classes,lpv) = log_proba(dfvalid,vec,cla)
(classes,lpt) = log_proba(dftest,vec,cla)
for i,k in enumerate(classes):
    j = cat1toi[k]
    stage1_log_proba_valid[:,j] = lpv[:,i]
    stage1_log_proba_test[:,j] = lpt[:,i]

del labels,vec,cla


#######################
# stage 3 log proba filling
#######################
stage3_log_proba_valid = np.full(shape=(len(dfvalid),cat3count),fill_value = -666.,dtype = float)
stage3_log_proba_test = np.full(shape=(len(dftest),cat3count),fill_value = -666.,dtype = float)

for ii,cat in enumerate(itocat1):
    fname = ddir + 'joblib/stage3_'+str(cat)+ext
    print '-'*50
    print 'predicting',basename(fname),':',ii,'/',len(itocat1)
    if not isfile(fname): 
        continue
    (labels,vec,cla,) = joblib.load(fname)
    if len(labels)==1:
        k = labels[0]
        j = cat3toi[k]
        stage3_log_proba_valid[:,j] = 0
        stage3_log_proba_test[:,j] = 0
        continue
    (classes,lpv) = log_proba(dfvalid,vec,cla)
    (classes,lpt) = log_proba(dftest,vec,cla)
    for i,k in enumerate(classes):
        j = cat3toi[k]
        stage3_log_proba_valid[:,j] = lpv[:,i]
        stage3_log_proba_test[:,j] = lpt[:,i]
    del labels,vec,cla

print '>>> write stage1 & stage2 log_proba'
joblib.dump((stage1_log_proba_valid,stage3_log_proba_valid),ddir+'/joblib/log_proba_valid'+ext)
joblib.dump((stage1_log_proba_test,stage3_log_proba_test),ddir+'/joblib/log_proba_test'+ext)
print '<<< write stage1 & stage2 log_proba'

##################
# (stage1_log_proba_valid,stage3_log_proba_valid) = joblib.load(ddir+'/joblib/log_proba_valid'+ext)
# (stage1_log_proba_test,stage3_log_proba_test) = joblib.load(ddir+'/joblib/log_proba_test'+ext)
##################

##################
# bayes rulez ....
##################

def greedy_prediction(stage1_log_proba,stage3_log_proba):
    cat1 = [itocat1[c] for c in stage1_log_proba.argmax(axis=1)]
    for i in range(stage3_log_proba.shape[0]):
        stage3_log_proba[i,:] = [stage3_log_proba[i,j] if cat3tocat1[cat3]==cat1[i] else -666 for j,cat3 in enumerate(itocat3)]
    return

def bayes_prediction(stage1_log_proba,stage3_log_proba):
    for i in range(stage3_log_proba.shape[1]):
        cat3 = itocat3[i]
        cat1 = cat3tocat1[cat3]
        j = cat1toi[cat1]
        stage3_log_proba[:,i] += stage1_log_proba[:,j]


assert stage3_log_proba_valid.shape[1] == stage3_log_proba_test.shape[1]

bayes_prediction(stage1_log_proba_valid,stage3_log_proba_valid)
bayes_prediction(stage1_log_proba_test,stage3_log_proba_test)

predict_cat1_valid = [itocat1[i] for i in np.argmax(stage1_log_proba_valid,axis=1)]
predict_cat3_valid = [itocat3[i] for i in np.argmax(stage3_log_proba_valid,axis=1)]
predict_cat1_test = [itocat1[i] for i in np.argmax(stage1_log_proba_test,axis=1)]
predict_cat3_test = [itocat3[i] for i in np.argmax(stage3_log_proba_test,axis=1)]

def submit(df,Y):
    submit_file = ddir+'resultat.auto'+ext+'.csv'
    df['Id_Produit']=df['Identifiant_Produit']
    df['Id_Categorie'] = Y
    df= df[['Id_Produit','Id_Categorie']]
    df.to_csv(submit_file,sep=';',index=False)

def save_proba(df,Y,p1,p3):
    submit_file = ddir+'proba.auto'+ext+'.csv'
    df['Id_Produit']=df['Identifiant_Produit']
    df['Id_Categorie'] = Y
    df['Proba_Categorie1'] = p1
    df['Proba_Categorie3'] = p3
    df= df[['Id_Produit','Id_Categorie','Proba_Categorie1','Proba_Categorie3']]
    df.to_csv(submit_file,sep=';',index=False)


score_cat1 = sum(dfvalid.Categorie1 == predict_cat1_valid)*1.0/len(dfvalid)
score_cat3 = sum(dfvalid.Categorie3 == predict_cat3_valid)*1.0/len(dfvalid)
print '#######################################'
print '# validation score :',score_cat1,score_cat3
print '#######################################'

submit(dftest,predict_cat3_test)

proba_cat1_test =  np.exp(np.max(stage1_log_proba_test,axis=1))
proba_cat3_test =  np.exp(np.max(stage3_log_proba_test,axis=1))

dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
save_proba(dftest,predict_cat3_test,proba_cat1_test,proba_cat3_test)

print '-'*50
