from utils import ddir,normalize_file

normalize_file(ddir + 'test.csv',header(test=True))
normalize_file(ddir + 'training_shuffled.csv',header())
