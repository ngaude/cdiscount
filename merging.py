#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: ngaude
"""

from utils import header,add_txt
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from utils import itocat1,itocat3
from utils import cat1count,cat2count,cat3count
import time

ddir = '/home/ngaude/workspace/data/cdiscount.proba/' 

dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
add_txt(dftest)
dftest = dftest[['Identifiant_Produit','txt']]

stage3_proba_test = np.full(shape=(len(dftest),cat3count),fill_value = 0.,dtype = float)
stage1_proba_test = np.full(shape=(len(dftest),cat1count),fill_value = 0.,dtype = float)

def submit(df,Y):
    submit_file = ddir+'resultat.auto.merging.'+str(N)+'.csv'
    df['Id_Produit']=df['Identifiant_Produit']
    df['Id_Categorie'] = Y
    df= df[['Id_Produit','Id_Categorie']]
    df.to_csv(submit_file,sep=';',index=False)

def save_proba(df,Y,p1,p3):
    submit_file = ddir+'proba.auto.merging.'+str(N)+'.csv'
    df['Id_Produit']=df['Identifiant_Produit']
    df['Id_Categorie'] = Y
    df['Proba_Categorie1'] = p1
    df['Proba_Categorie3'] = p3
    df= df[['Id_Produit','Id_Categorie','Proba_Categorie1','Proba_Categorie3']]
    df.to_csv(submit_file,sep=';',index=False)

proba_files = [
#    ddir+'joblib/proba_test_stacked.100-116',
#    ddir+'joblib/proba_test_stacked.200-220',
#    ddir+'joblib/proba_test_stacked.300-311',
#    ddir+'joblib/proba_test_stacked.0-19',
    ddir+'joblib/proba_test_stacked.0-4',
    ]

N=0

for f in proba_files:
    print '>> merging ',f
    (l,s1,s3) = joblib.load(f)
    stage1_proba_test += s1
    stage3_proba_test += s3
    N += len(l)
    del s1,s3

stage1_proba_test /= N
stage3_proba_test /= N

proba_cat1_test =  (np.max(stage1_proba_test,axis=1))
proba_cat3_test =  (np.max(stage3_proba_test,axis=1))
predict_cat1_test = [itocat1[i] for i in np.argmax(stage1_proba_test,axis=1)]
predict_cat3_test = [itocat3[i] for i in np.argmax(stage3_proba_test,axis=1)]

submit(dftest,predict_cat3_test)

dftest = pd.read_csv(ddir+'test_normed.csv',sep=';',names = header(test=True)).fillna('')
save_proba(dftest,predict_cat3_test,proba_cat1_test,proba_cat3_test)

