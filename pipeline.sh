#!/bin/sh

DDIR=/home/ngaude/workspace/data/cdiscount/
mkdir $DDIR/joblib


# shuffle training set : 5mn
sed 1d $DDIR/training.csv | shuf > $DDIR/training_shuffled.csv

# normalize test and training set : ~ 1h
time python normalizing.py

# split training set into train & validation : 2mn
head -n 15500000 $DDIR/training_shuffled_normed.csv > $DDIR/training_head.csv
tail -n 286885 $DDIR/training_shuffled_normed.csv > $DDIR/training_tail.csv

# sample and train 5 pyramids : 5 x 2h
time python sampling.py 0 
time python training.py 0
time python sampling.py 1
time python training.py 1
time python sampling.py 2
time python training.py 2
time python sampling.py 3
time python training.py 3
time python sampling.py 4
time python training.py 4

# stack & merge results : 5mn
time python stacking.py
time python merging.py
