import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from fnn import FNNClassifier
class featureextractor():

        def extract_features(self,sents,stoplist,stoppunc,ngram=1,vocab=None,lowercase=True,stopwords=True,nfeats=100,min_df=1,lemmatize=False):


                stop = (set.union(stoppunc,stoplist) if stopwords else stoppunc)
                #sents = self.lemmatize(sents) if lemmatize==True else sents
                #print("stop->",stop)
                vectorizer = TfidfVectorizer(lowercase=lowercase,stop_words=stop,vocabulary=vocab,min_df=min_df,max_features=None,ngram_range=(1,ngram))
                csrmatrix = vectorizer.fit_transform(sents)
                self.X = csrmatrix
                self.mapping = vectorizer.get_feature_names()
                self.vocab = vectorizer.vocabulary_
                self.vectorizer = vectorizer
                return csrmatrix.toarray()

        def get_features(self,sents):
                csrmatrix = self.vectorizer.transform(sents)
                return csrmatrix.toarray()

class models():
	def __init__(self):
		noth=1
		#trainextra = pd.read_csv("../datasets/train_in_posfeatures.csv")
		#del trainextra["id"]
		#self.trainextra = trainextra.as_matrix()
		#testextra = pd.read_csv("../datasets/test_in_posfeatures.csv")
		#del testextra["id"]		
		#self.testextra = testextra.as_matrix()
	
	def kfoldCV(self,X,Y,modelname,classes,loops):
		k=5
		datasets = np.array_split(X,k,axis=0)
		testsets = np.array_split(Y,k,axis=0) 
		#extradata = np.array_split(self.trainextra,k,axis=0)
		CM_model=np.zeros((classes,classes)).astype(float)
		model = eval(modelname)
		self.modelname = modelname
		error=float(0)
		nooft=0
		noofv=0
		for i in range(0,k):
#generate datasets
			learnon = np.concatenate((datasets[i % k], datasets[(i + 1) % k], datasets[(i + 2) % k], datasets[(i + 3) % k]),
                             axis=0)
			learny = np.concatenate((testsets[i % k], testsets[(i + 1) % k], testsets[(i + 2) % k], testsets[(i + 3) % k]),
                            axis=0)
			teston = datasets[(i + 4) % k]
			testy = testsets[(i + 4) % k]
			#extratest = extradata[(i+4)%k]
			
			model.fit(learnon,learny)
			#mysvm=svm.svm()
			#mysvm.fit(learnon,learny)
			#print(model.n_support_)
			#print("examples",learnon.shape[0])
#make predictions
			model_pred=model.predict(teston)
			model_trainpred = model.predict(learnon)
			for m in range(0,testy.shape[0]):
				CM_model[testy[m]][model_pred[m]]=CM_model[testy[m]][model_pred[m]]+1
				noofv = noofv+1	
			for m in range(0,learny.shape[0]):
				if(learny[m]!=model_trainpred[m]):
					error = error+1
				nooft=nooft+1
			if loops==0:
				break
			print("Validation Stage: ",k)
#calculate errors
		recall=[]
		precision=[]
		f1=[]
		for i in range(0,classes):
			ignore=False
			if(float(np.sum(CM_model[:,i]))>0):
				prec = float(CM_model[i][i])/float(np.sum(CM_model[:,i]))
				precision.append(prec)
			else:
				ignore=True
			if(float(np.sum(CM_model[i]))>0):
				rec=float(CM_model[i][i])/float(np.sum(CM_model[i]))
				recall.append(rec)
			else:
				ignore=True
			if(ignore is False):
				if((prec+rec)>0):
					f1.append(2*prec*rec/(prec+rec))
				else:
					f1.append(0)
		error_valid = 1-np.trace(CM_model)/noofv
		error_train = error/nooft
		self.trainingError = error_train
		self.validationError = error_valid
		#print(CM_model)
		self.CM = CM_model
		return error_train,error_valid,precision,recall,f1
	def setpred(self,XP):
		self.XP=XP
	def predict(self,X,Y,XP,modelname):
		model = eval(modelname)
		model.fit(X,Y)
		return model.predict(XP)
