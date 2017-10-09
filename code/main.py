import numpy as np
from nltk.corpus import stopwords
import string
import support
import models as md
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA, FactorAnalysis

#change below boolean variable for SIFT
usesift=True


classes=19
X, Y, idstrain = support.read_data('train',sift=usesift)
X_test = support.read_data('test')
X,Y = shuffle(X,Y,random_state=17)
print(Y)
print(X.shape)
print(Y.shape)
print(X_test.shape)
#sentences = (support.lemmatize(sentences))
#support.write_lemma(sentences,idstrain,"train")
#print("Training lemmatization done")
#sentences_test = support.lemmatize(sentences_test)
#print("Testing Lemmatization done")
#support.write_lemma(sentences_test,ids,"test")
#make this true to generate plots
plot=True
models = []
#Explanation for models respectively 
# Name of the Graph or file, unigram(1) or unigram and bigram(2), convert to lowercase boolean, include stop words boolean, Number of features, run only a single cross validation iteration (for time speedup), boolean for writing test_out.csv, class name for creating the object

models.append(["Logistic Regression",X.shape[1],"LogisticRegression(C=1000)",False])
models.append(["SVM",X.shape[1],"svm.SVC()",False])
hiddenl = "(100"
for h in range(1,40):
	hiddenl = hiddenl + ",100"
hiddenl = hiddenl + ")"
models.append(["Neural Net Dropout=0.25, Nodes=40, HL=40",X.shape[1],"FNNClassifier(num_of_hidden_layers=40, hidden_layer_sizes="+hiddenl+",do_dropout=True, dropout_rate=0.25, learning_rate=0.01)",False])

for i in range(0,len(models)):
	terrors=[]
	verrors=[]
	feats=[]
	precisions=[]
	f1s=[]
	recalls=[]
	minfeat=1
	maxfeat=6
	lr=0.1
	step=1
	for j in range(minfeat,maxfeat,step):
		#hiddenl = "(100"
		#for h in range(1,j):
		#	hiddenl = hiddenl + ",100"
		#hiddenl = hiddenl + ")"
		model=md.models()
		Terror,Verror,precision,recall,f1 = model.kfoldCV(X,Y,models[i][2],classes,0)
		print(models[i])
		print("Training Error = ",Terror)
		print("Validation Error = ",Verror)
		#precision=np.array(precision)
		#recall=np.array(recall)		
		print("Avg Precision",np.mean(precision))
		print("Avg Recall",np.mean(recall))
		print("Avg F1",np.mean(f1))
		terrors.append(Terror)
		verrors.append(Verror)
		precisions.append(precision)
		recalls.append(recall)
		f1s.append(f1)
		models[i][1]=lr
		feats.append(lr)
		lr=lr/float(10)
#generate tes result or not
		if models[i][3]:
			predictions=model.predict(X,Y,X_test,models[i][2])
			support.write_data(predictions)	
	if plot:
		#validation error plot
		plt.plot(feats, terrors)
		plt.plot(feats,verrors)
		plt.xlabel("Learning Rate")
		plt.ylabel("Error")
		plt.legend(['Training', 'Validation'], loc='upper right')
		plt.title(models[i][0])
		plt.savefig("../Fig/"+models[i][0]+'_lr.png', format='png')
		plt.clf()
		'''
		precisions = np.array(precisions)
		recalls = np.array(recalls)
		f1s = np.array(f1s)
		#precision recall plot
		plt.plot(precisions[:,0].tolist(),recalls[:,0].tolist())
		plt.plot(precisions[:,1].tolist(),recalls[:,1].tolist())
		plt.plot(precisions[:,2].tolist(),recalls[:,2].tolist())
		plt.plot(precisions[:,3].tolist(),recalls[:,3].tolist())
		plt.legend(['math',"cs","stat", 'physics'], loc='lower right')
		plt.xlabel("Recall")
		plt.ylabel("Precision")
		plt.title(models[i][0])
		plt.savefig("../Fig/"+models[i][0]+'_precall.png', format='png')
		plt.clf()
		#F1 plot
		plt.plot(feats,f1s[:,0])
		plt.plot(feats,f1s[:,1])
		plt.plot(feats,f1s[:,2])
		plt.plot(feats,f1s[:,3])
		plt.legend(['math',"cs","stat", 'physics'], loc='lower left')
		plt.xlabel("Number of Features")
		plt.ylabel("F1 Score")
		plt.title(models[i][0])
		plt.savefig("../Fig/"+models[i][0]+'_f1.png', format='png')
		plt.clf()
		'''
		#write to file
		data=pd.DataFrame([feats,terrors,verrors]).transpose()
		data.columns = ["Number of Nodes","Training Error","Validation Error"]
		#precisions=pd.DataFrame(precisions)
		#precisions.columns = ["p math","p cs","p stat","p physics"]
		#recalls=pd.DataFrame(recalls)
		#recalls.columns = ["r math","r cs","r stat","r physics"]
		#f1s=pd.DataFrame(f1s)
		#f1s.columns = ["f1 math","f1 cs","f1 stat","f1 physics"]
		#data=pd.concat([data,precisions,recalls,f1s],axis=1)
		data.to_csv("../Fig/"+models[i][0]+"_lr.csv")

	
