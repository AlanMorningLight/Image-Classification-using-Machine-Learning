import sys
import numpy
import scipy.misc # to visualize only
import pandas as pd
#import cv2
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import time
noof_train=100000
noof_test = 20000
xtrain = '../Data/train_x.bin'
xtest = '../Data/test_x.bin'


def base_extractor(data,samples):
	x = numpy.fromfile(data, dtype='uint8')
	x = x.reshape((samples,60,60))
	y = x.flatten()
	y = y.reshape(samples,3600)
	y=y/255
	#y=y.astype(int)
	return y
def sift_extractor(data,samples):
	x = numpy.fromfile(data, dtype='uint8')
	x = x.reshape((samples,60,60))
	#ytrain = '../Data/train_y.csv'
	#y=pd.read_csv(ytrain)
	#y=y["Prediction"].as_matrix()
	#x,y=shuffle(x,y,random_state=53)
	#xlearn = x[:20000]
	#xrest = x[20000:]
	#y = y[20000:]
	sift = cv2.SIFT()
	X=[]
	kmeans = joblib.load('../Data/kmeansmodel_50.pkl') 
	for i in range(0,20000):
		t1=time.time()
		kp = sift.detect(x[i],None)
		kp,des = sift.compute(x[i],kp)
		if(des is None):
			#y=np.delete(y,(i-20000))
			print("Values is null")
			continue
		results=kmeans.predict(des)
		tmp = numpy.zeros(50)
		for k in range(0,len(results)):
			tmp[results[k]] += 1 
		X.append(tmp)
		t2=time.time()
		timeleft = float(20000-i-1)*(t2-t1)
		progress=float(i+1)/float(200)
		sys.stdout.write("Conversion progress: %f%% Time Remaining: %d  \r" % (progress,timeleft) )
		sys.stdout.flush()
	X=pd.DataFrame(X)
	#Y=pd.DataFrame(y)
	X.to_csv('../Data/testfeaturessift.csv',index=False,header=False)
	#Y.to_csv('../Data/ysift.csv',index=False,header=False)

def sift_extractor_save(data,samples):
	x = numpy.fromfile(data, dtype='uint8')
	x = x.reshape((samples,60,60))
	ytrain = '../Data/train_y.csv'
	y=pd.read_csv(ytrain)
	y=y["Prediction"].as_matrix()
	x,y=shuffle(x,y,random_state=53)
	xlearn = x[:20000]
	xrest = x[20000:]
	y = x[20000:]
	sift = cv2.SIFT()
	kp = sift.detect(x[0],None)
	kp,des = sift.compute(x[0],kp)
	clusterdata = numpy.copy(des)
	print("about to start reading data")
	for i in range(1,20000):
		kp = sift.detect(x[i],None)
		kp,des = sift.compute(x[i],kp)
		if(des is None):
			print("received null")
			continue
		clusterdata = numpy.concatenate((clusterdata,numpy.copy(des)),axis=0)
	print("starting to learn model")
	kmeans_model = KMeans(n_clusters=50, random_state=1).fit(clusterdata)
	print("model finished learning")
	joblib.dump(kmeans_model, '../Data/kmeansmodel_50.pkl') 

#x = numpy.fromfile(xtest, dtype='uint8')
#x = x.reshape((20000,60,60))
#sift = cv2.SIFT()
#kp = sift.detect(x[1],None)
#kp,des = sift.compute(x[0],kp)
#img=cv2.drawKeypoints(x[1],kp)
#cv2.imwrite('sift_keypoints.jpg',img)
#sift_extractor_save(xtrain,100000)
#sift_extractor(xtest,20000)
'''
x=base_extractor(xtrain,noof_train)
x=pd.DataFrame(x)
x.to_csv("../Data/train_features.csv",header=False)
del (x)
x=base_extractor(xtest,noof_test)
x=pd.DataFrame(x)
x.to_csv("../Data/test_features.csv",header=False)
'''
