#from itertools import izip
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import feature_extraction


def read_data(mode,sift=False):
	noof_train=100000
	noof_test = 20000
	xtrain = '../Data/train_x.bin'
	ytrain = '../Data/train_y.csv'
	xtest = '../Data/test_x.bin'
	if(mode=="train"):
		if(sift==True):
			Y=pd.read_csv('../Data/ysift.csv',header=None)
			X=pd.read_csv('../Data/featuressift.csv',header=None)
			Y=Y.as_matrix()
			X=X.as_matrix()
			return X, np.ravel(Y), None
		x=feature_extraction.base_extractor(xtrain,noof_train)
		ypath = '../Data/train_y.csv'
		Y=pd.read_csv(ypath)
		ids=Y["Id"].as_matrix()
		Y=Y["Prediction"].as_matrix()
		return x, Y, ids
	else: 
		x=feature_extraction.base_extractor(xtest,noof_test)
		return x					
    		
		
		

def write_data(pred):
	filepath = '../Data/test_out.csv'
	data = pd.DataFrame([pred]).transpose()
	data.columns = ["Predictions"]
	#data["Predictions"] = data["Predictions"]
	data.to_csv(filepath)

def write_lemma(sents,ids,mode):
	if mode=="train":
		filepath = '../datasets/train_in_lemma.csv'
	else:
		filepath = '../datasets/test_in_lemma.csv'
	data = pd.DataFrame([ids,sents]).transpose()
	data.columns = ["id","abstract"]
	data.to_csv(filepath,index=False)

def write_pos(sents,ids,mode):
        if mode=="train":
                filepath = '../datasets/train_in_pos.csv'
        else:
                filepath = '../datasets/test_in_pos.csv'
        data = pd.DataFrame([ids,sents]).transpose()
        data.columns = ["id","abstract"]
        data.to_csv(filepath,index=False)
