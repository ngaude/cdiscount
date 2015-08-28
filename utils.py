import marisa_trie
from sklearn.externals import six
from sklearn.feature_extraction.text import TfidfVectorizer
#import Stemmer # NOTE : pip install pyStemmer
import nltk
from bs4 import BeautifulSoup
import re
import unicodedata 
import time
import pandas as pd
import numpy as np
import random
import os
from sklearn.neighbors import NearestNeighbors
from scipy import sparse


ddir = '/home/ngaude/workspace/data/cdiscount/'
wdir = '/home/ngaude/workspace/github/cdiscount/'

rayon = pd.read_csv(ddir+'rayon.csv',sep=';')

itocat1 = list(np.unique(rayon.Categorie1))
cat1toi = {cat1:i for i,cat1 in enumerate(itocat1)}
itocat2 = list(np.unique(rayon.Categorie2))
cat2toi = {cat2:i for i,cat2 in enumerate(itocat2)}
itocat3 = list(np.unique(rayon.Categorie3))
cat3toi = {cat3:i for i,cat3 in enumerate(itocat3)}
cat3tocat2 = rayon.set_index('Categorie3').Categorie2.to_dict()
cat3tocat1 = rayon.set_index('Categorie3').Categorie1.to_dict()
cat2tocat1 = rayon[['Categorie2','Categorie1']].drop_duplicates().set_index('Categorie2').Categorie1.to_dict()
cat1count = len(np.unique(rayon.Categorie1))
cat2count = len(np.unique(rayon.Categorie2))
cat3count = len(np.unique(rayon.Categorie3))

stopwords = []
with open(wdir+'stop-words_french_1_fr.txt', "r") as f:
    stopwords += f.read().split('\n')

with open(wdir+'stop-words_french_2_fr.txt', "r") as f:
    stopwords += f.read().split('\n')

stopwords += nltk.corpus.stopwords.words('french')
stopwords += ['voir', 'presentation']
stopwords = set(stopwords)

#stemmer = Stemmer.Stemmer('french')
stemmer=nltk.stem.SnowballStemmer('french')

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def header(test=False):
    if test==True:
        columns = ['Identifiant_Produit','Description','Libelle','Marque','prix']
    else:
        columns = ['Identifiant_Produit','Categorie1','Categorie2','Categorie3','Description','Libelle','Marque','Produit_Cdiscount','prix']
    return columns

def normalize_guess(txt):
    ####################################
    # use for brute force wild guessing
    # NOTE : no stemming for wild guess
    ####################################
    # remove html stuff
    txt = BeautifulSoup(txt,from_encoding='utf-8').get_text()
    # lower case
    txt = txt.lower()
    # special escaping character '...'
    txt = txt.replace(u'\u2026','.')
    txt = txt.replace(u'\u00a0',' ')
    # remove accent btw
    txt = unicodedata.normalize('NFD', txt).encode('ascii', 'ignore')
    #txt = unidecode(txt)
    # remove non alphanumeric char
    txt = re.sub('[^a-z_]', ' ', txt)
    # remove french stop words
    tokens = [w for w in txt.split() if (len(w)>2) and (w not in stopwords)]
    # french stemming
    return ' '.join(tokens)


def normalize_txt(txt):
    # remove html stuff
    txt = BeautifulSoup(txt,from_encoding='utf-8').get_text()
    # lower case
    txt = txt.lower()
    # special escaping character '...'
    txt = txt.replace(u'\u2026','.')
    txt = txt.replace(u'\u00a0',' ')
    # remove accent btw
    txt = unicodedata.normalize('NFD', txt).encode('ascii', 'ignore')
    #txt = unidecode(txt)
    # remove non alphanumeric char
    txt = re.sub('[^a-z_]', ' ', txt)
    # remove french stop words
    tokens = [w for w in txt.split() if (len(w)>2) and (w not in stopwords)]
    # french stemming
    tokens = [ stemmer.stem(token) for token in tokens]
#    tokens = stemmer.stemWords(tokens)
    return ' '.join(tokens)

def normalize_price(price):
    if (price<0) or (price>100):
        price = 0
    return price

def normalize_file(fname,header,nrows = None):
    columns = {k:v for v,k in enumerate(header)}
    ofname = fname.split('.')[0]+'_normed.'+fname.split('.')[1]
    ff = open(ofname,'w')
    start_time = time.time()
    counter = 0
    for line in open(fname):
        if line.startswith('Identifiant_Produit'):
            continue
        di = columns['Description']
        li = columns['Libelle']
        mi = columns['Marque']
        pi = columns['prix']
        if counter%1000 == 0:
            print fname,': lines=',counter,'time=',int(time.time() - start_time),'s'
        ls = line.split(';')
        # marque normalization
        txt = ls[mi]
        txt = re.sub('[^a-zA-Z0-9]', '_', txt).lower()
        ls[mi] = txt
        #
        # description normalization
        ls[di] = normalize_txt(ls[di])
        #
        # libelle normalization
        ls[li] = normalize_txt(ls[li])
        #
        # prix normalization
        ls[pi] = str(normalize_price(float(ls[pi].strip())))
        line = ';'.join(ls)
        ff.write(line+'\n')
        counter += 1
        if (nrows is not None) and (counter>=nrows):
            break
    ff.close()
    return

class iterText(object):
    def __init__(self, df):
        """
        Yield each document in turn, as a text.
        """
        self.df = df
    
    def __iter__(self):
        for row_index, row in self.df.iterrows():
            if (row_index>0) and (row_index%10000)==0:
                print row_index,'/',len(self.df)
            txt = ' '.join([row.Marque]*3+[row.Libelle]*2+[row.Description])
            yield txt
    
    def __len__(self):
        return len(self.df)

class MarisaTfidfVectorizer(TfidfVectorizer):
    def fit_transform(self, raw_documents, y=None):
        super(MarisaTfidfVectorizer, self).fit_transform(raw_documents)
        self._freeze_vocabulary()
        return super(MarisaTfidfVectorizer, self).fit_transform(raw_documents, y)
    def fit(self, raw_documents, y=None):
        super(MarisaTfidfVectorizer, self).fit(raw_documents)
        self._freeze_vocabulary()
        return super(MarisaTfidfVectorizer, self).fit(raw_documents, y)
    def _freeze_vocabulary(self, X=None):
        if not self.fixed_vocabulary_:
            self.vocabulary_ = marisa_trie.Trie(six.iterkeys(self.vocabulary_))
            self.fixed_vocabulary_ = True
            del self.stop_words_


def training_sample(dftrain,label,mincount = 200,maxsampling = 10):
    cl = dftrain[label]
    cc = cl.groupby(cl)
    s = (cc.count() > mincount/maxsampling)
    labelmaj = s[s].index
    print len(labelmaj),len(labelmaj)*mincount
    dfs = []
    for i,cat in enumerate(labelmaj):
        if i%10==0:
            print i,'/',len(labelmaj),':'
        df = dftrain[dftrain[label] == cat]
        if len(df)>=mincount:
            # undersample mincount samples
            rows = random.sample(df.index, mincount)
            dfs.append(df.ix[rows])
        else:
            # sample all samples + oversample the remaining
            dfs.append(df)
            df = df.iloc[np.random.randint(0, len(df), size=mincount-len(df))]
            dfs.append(df)
    dfsample = pd.concat(dfs)
    dfsample = dfsample.reset_index(drop=True)
    dfsample = dfsample.reindex(np.random.permutation(dfsample.index),copy=False)
    return dfsample

def add_txt(df):
    assert 'Marque' in df.columns
    assert 'Libelle' in df.columns
    assert 'Description' in df.columns
    df['txt'] = get_txt(df)
    return

def get_txt(df):
    assert 'Marque' in df.columns
    assert 'Libelle' in df.columns
    assert 'Description' in df.columns
    return (df.Marque+' ')*3+(df.Libelle+' ')*2+df.Description

def result_diffing(fx,fy):
    dfx = pd.read_csv(fx,sep=';').fillna('')
    dfy = pd.read_csv(fy,sep=';').fillna('')
    test = pd.read_csv(ddir+'test.csv',sep=';').fillna('')
    rayon = pd.read_csv(ddir+'rayon.csv',sep=';').fillna('')
    dfx = dfx.merge(rayon,'left',None,'Id_Categorie','Categorie3')
    dfy = dfy.merge(rayon,'left',None,'Id_Categorie','Categorie3')
    dfx = dfx.merge(test,'left',None,'Id_Produit','Identifiant_Produit')
    df = dfx.merge(dfy,'inner','Id_Produit')
    df = df[df.Categorie3_x != df.Categorie3_y]
    df = df[['Id_Produit','Categorie3_Name_x','Categorie3_Name_y','Marque','Libelle','Description','prix']]
    df.to_csv(ddir+'diff.csv',sep=';',index=False)
    return df


